import inspect
import numpy as np
from tqdm import tqdm
import torch
import torch.cuda.amp as amp

from utils import HuggingfaceTokenizer
from utils.prompt_extend import call_with_messages
from utils.utils import TensorList, to_, cache_video
from utils.fm_solvers import FlowDPMSolverMultistepScheduler
import models
from models.vae_mulshot import VideoVAE
from models.model_mulshot_rope_dual import Transformer
import os
import torch.distributed as dist
import re
from einops import rearrange

import json

class T5Encoder:

    def __init__(
        self,
        name,
        text_len,
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),
        checkpoint_path=None,
        tokenizer_path=None,
    ):
        self.name = name
        self.text_len = text_len
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        # init model
        model = getattr(models, name)(
            encoder_only=True,
            return_tokenizer=False,
            dtype=dtype,
            device=device
        ).eval().requires_grad_(False)
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.model = model

        # init tokenizer
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path,
            seq_len=text_len,
            clean='whitespace'
        )
    
    def __call__(self, texts):
        ids, mask = to_(self.tokenizer(
            texts,
            return_mask=True,
            add_special_tokens=True
        ), self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.model(ids, mask)
        return [u[:v] for u, v in zip(context, seq_lens)]

def get_sampling_sigmas(sampling_steps, shift):
    sigma = np.linspace(1, 0, sampling_steps+1)[:sampling_steps]
    sigma = (shift * sigma / (1 + (shift - 1) * sigma))

    return sigma


def retrieve_timesteps(
    scheduler,
    num_inference_steps= None,
    device= None,
    timesteps= None,
    sigmas = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def worker(gpu, cfg):

    torch.cuda.set_device(gpu)
    rank = gpu

    dist.init_process_group(
        backend='nccl',  # modern: 'cpu:gloo,cuda:nccl'
        rank=rank,
        world_size=cfg.world_size,
    )
    device = 'cuda'
    # [model] t5
    t5 = T5Encoder(
        name=cfg.t5_model,
        text_len=cfg.text_len,
        dtype=cfg.t5_dtype,
        device=device,
        checkpoint_path=cfg.t5_checkpoint,
        tokenizer_path=cfg.t5_tokenizer
    )

    # [model] vae
    vae = VideoVAE(
        vae_pth=cfg.vae_checkpoint,
        device=device
    )

    # [model] transformer
    model = Transformer(
        patch_size=cfg.patch_size,
        text_len=cfg.text_len,
        in_dim=vae.model.z_dim,
        dim=cfg.dim,
        ffn_dim=cfg.ffn_dim,
        freq_dim=cfg.freq_dim,
        text_dim=t5.model.dim,
        out_dim=vae.model.z_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        window_size=cfg.window_size,
        qk_norm=cfg.qk_norm,
        cross_attn_norm=cfg.cross_attn_norm,
        eps=cfg.eps
    ).requires_grad_(False)

    state_dict = torch.load(cfg.checkpoint_path, map_location='cpu')
    status = model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()

    # clear cache
    torch.cuda.empty_cache()

    ###
    os.makedirs(cfg.out_dir, exist_ok=True)
    pattern = r"step_(\d{3,5})"
    matches = re.findall(pattern, cfg.checkpoint_path)
    if len(matches) == 0:
        ckpt = 'original'
    else:
        ckpt = 'ckpt' + matches[0]
    out_name = cfg.out_dir + '/' + ckpt + f'_seed{cfg.base_seed}_' + cfg.postfix + '_' + cfg.out_dir.split('/')[-1]

    assert isinstance(cfg.input_prompts, str) and cfg.input_prompts.endswith('json'), 'Prompts should be a path of json file.'
    with open(cfg.input_prompts, 'r') as file:
        input = json.load(file)
    cfg.input_prompts = input['cap_list']
    cfg.inner_t = input['inner_t']
    
    ind_list = [i for i in range(len(cfg.input_prompts))][rank::cfg.world_size]
    cfg.input_prompts = [cfg.input_prompts[i] for i in ind_list]
    
    
    t = cfg.target_shape[1]
    t0 = sum(cfg.inner_t[0])
    deltat = t-t0
    if deltat==0:
        cfg.inner_t = [cfg.inner_t[i] for i in ind_list]
    else:
        inner_t = []
        for i in ind_list:
            t_ = cfg.inner_t[i]
            tplus = [(t-t0)//len(t_)]*(len(t_)-1) + [(t-t0)-((t-t0)//len(t_)*(len(t_)-1))]
            inner_t.append([m+n for _, (m,n) in enumerate(zip(t_, tplus))])
        cfg.inner_t = inner_t
                  
    
    if cfg.use_prompt_extend:
        input_prompts = [call_with_messages(p, None, None, 0).replace('\n', '\\n') for p in cfg.input_prompts]
        print('enable extended prompt: ',input_prompts)
    else:
        input_prompts = cfg.input_prompts

    # preprocess
    context = []
    context_null = []
    null = t5([cfg.sample_neg_prompt])
    for input in input_prompts:
        context.append(t5(input))
        context_null.append(null*len(input))
    t5 = t5.model.to('cpu')

    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(cfg.base_seed+gpu)
    
    noise = [torch.randn(cfg.target_shape[0], cfg.target_shape[1], cfg.target_shape[2], cfg.target_shape[3], dtype=torch.float32, device=device, generator=seed_g) for _ in range(len(input_prompts))]
    
    # evaluation mode
    with (
        amp.autocast(dtype=cfg.param_dtype),
        torch.no_grad(),
    ):
        sample_scheduler = FlowDPMSolverMultistepScheduler(num_train_timesteps=cfg.num_train_timesteps, shift=1, use_dynamic_shifting=False)
        
        sampling_sigmas = get_sampling_sigmas(cfg.sample_steps, cfg.sample_shift)

        # sample videos
        latents = noise

        for j in range(len(latents)):

            timesteps, _ = retrieve_timesteps(sample_scheduler, device=device, sigmas=sampling_sigmas)
            arg_c = {'context': context[j:j+1], 'seq_len': cfg.max_seq_len, 'inner_t': cfg.inner_t[j:j+1]}
            arg_null = {'context': context_null[j:j+1], 'seq_len': cfg.max_seq_len, 'inner_t': cfg.inner_t[j:j+1]}

            for i, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents[j:j+1]
                timestep = [t]

                timestep = torch.stack(timestep)

                noise_pred_cond = TensorList(model(TensorList(latent_model_input), t=timestep, **arg_c))
                noise_pred_uncond = TensorList(model(TensorList(latent_model_input), t=timestep, **arg_null))

                noise_pred = noise_pred_uncond + cfg.sample_guide_scale * (noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(noise_pred[0].unsqueeze(0), t, latents[j].unsqueeze(0), return_dict=False, generator=seed_g)[0]
                latents[j] = temp_x0.squeeze(0)

            x0 = latents[j]    # list

            fake_video = vae.decode([x0], inner_t=cfg.inner_t[j:j+1])    # video list
            # fake_video = [rearrange(fake_video[0], 's c t h w -> c t h (s w)')] ### display all shots
            save_file = out_name + f'_{ind_list[j]}.mp4'
            cache_video(
                tensor=fake_video[0].unsqueeze(0),
                save_file=save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1)
            )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--postfix", type=str)
    parser.add_argument("--type", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    from config_inference import cfg
    
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    gpus = torch.cuda.device_count()
    print(f'rank:{rank} world size:{world_size} gpus:{gpus}')
    cfg.checkpoint_path = args.ckpt
    cfg.input_prompts = args.prompt
    cfg.postfix = args.postfix
    cfg.out_dir = args.out_dir
    cfg.world_size = world_size * gpus
    cfg.debug = args.debug
    cfg.base_seed = args.seed
    if args.debug:
        worker(0, cfg)
    else:
        torch.multiprocessing.spawn(worker, nprocs=gpus, args=(cfg, ))

    