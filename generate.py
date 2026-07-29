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
from models.vae import VideoVAE
from models.model import Transformer
import os
import re
from datetime import datetime


def process_input(prompt, shot_len, total_t):
    segments = re.split(r'\[\d+\]', prompt)
    prompt = [segment.strip() for segment in segments if segment.strip()]
    assert len(prompt) > 1 and len(prompt) < 7, 'Input shot num must between 2~6 !'
    if shot_len != "":
        ts = shot_len.split(',')
        shot_len = [float(t.strip()) for t in ts]
        shot_len = [int(t/sum(shot_len)*total_t) for t in shot_len]
        shot_len[-1] = total_t - sum(shot_len[:-1])
        assert min(shot_len) > 0, 'Some shot lengths are too small, please give a proper input or unspecify it for auto.'
        assert len(prompt) == len(shot_len), 'Input num of prompts must equal input num of shot lens ! You can unspecify the shot len for auto.'
    else:
        shot_num = len(prompt)
        shot_len = [total_t//shot_num] * (shot_num-1)
        shot_len.append(total_t-sum(shot_len))
    
    return [prompt], [shot_len]
    

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


def gen(args):
    H = 480
    W = 832
    T = args.sample_frames
    t = (T-1)//4+1
    args.target_shape = (16, t, H//8, W//8)
    

    now = datetime.now()
    filename_time = now.strftime("%Y%m%d_%H%M%S")
    device = 'cuda'
    # [model] t5
    t5 = T5Encoder(
        name='umt5_xxl',
        text_len=512,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=args.t5_checkpoint,
        tokenizer_path=args.t5_tokenizer
    )

    # [model] vae
    vae = VideoVAE(
        vae_pth=args.vae_checkpoint,
        device=device
    )

    # [model] transformer
    model = Transformer(
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=vae.model.z_dim,
        dim=1536,
        ffn_dim=8960,
        freq_dim=256,
        text_dim=t5.model.dim,
        out_dim=vae.model.z_dim,
        num_heads=12,
        num_layers=30,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6
    ).requires_grad_(False)

    state_dict = torch.load(args.checkpoint_path, map_location='cpu')
    status = model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()

    # clear cache
    torch.cuda.empty_cache()


    if args.use_prompt_extend:
        args.prompt = call_with_messages(args.prompt)
    prompt, args.shot_len = process_input(args.prompt, args.shot_len, args.target_shape[1])


    os.makedirs(args.out_dir, exist_ok=True)
    out_name = os.path.join(args.out_dir, f'output_seed{args.base_seed}')

    # preprocess
    context = []
    context_null = []
    null = t5([args.sample_neg_prompt])
    for input in prompt:
        context.append(t5(input))
        context_null.append(null*len(input))
    t5 = t5.model.to('cpu')

    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(args.base_seed)
    
    noise = [torch.randn(args.target_shape[0], args.target_shape[1], args.target_shape[2], args.target_shape[3], dtype=torch.float32, device=device, generator=seed_g) for _ in range(args.sample_times)]
    
    # evaluation mode
    with (
        amp.autocast(dtype=torch.bfloat16),
        torch.no_grad(),
    ):
        sample_scheduler = FlowDPMSolverMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        
        sampling_sigmas = get_sampling_sigmas(args.sample_steps, args.sample_shift)

        # sample videos
        latents = noise
        arg_c = {'context': context, 'seq_len': t * W * H // 16 // 16, 'inner_t': args.shot_len}
        arg_null = {'context': context_null, 'seq_len': t * W * H // 16 // 16, 'inner_t': args.shot_len}
        for j in range(len(latents)):

            timesteps, _ = retrieve_timesteps(sample_scheduler, device=device, sigmas=sampling_sigmas)

            for i, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents[j:j+1]
                timestep = [t]

                timestep = torch.stack(timestep)

                noise_pred_cond = TensorList(model(TensorList(latent_model_input), t=timestep, **arg_c))
                noise_pred_uncond = TensorList(model(TensorList(latent_model_input), t=timestep, **arg_null))

                noise_pred = noise_pred_uncond + args.sample_guide_scale * (noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(noise_pred[0].unsqueeze(0), t, latents[j].unsqueeze(0), return_dict=False, generator=seed_g)[0]
                latents[j] = temp_x0.squeeze(0)

            x0 = latents[j]    # list

            fake_video = vae.decode([x0], inner_t=args.shot_len)    # video list

            save_file = out_name + '_' + filename_time + f'_{j}.mp4'
            cache_video(
                tensor=fake_video[0].unsqueeze(0),
                save_file=save_file,
                fps=args.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1)
            )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True,
                        help='Can be a string of shot prompts')
    parser.add_argument('--use_prompt_extend', action="store_true", default=False,
                        help='Using prompt extend is highly recommended!')
    parser.add_argument("--shot_len", type=str, default="",
                        help='Determine the length of each shot. Will be auto if not specified.')
    parser.add_argument('--sample_frames', type=int, default=93,
                        help='Total generation frames. Values between 93~125 are recommended (default: 93)')
    parser.add_argument('--sample_times', type=int, default=1,
                        help='Repeat times per generation (default: 1)')
    parser.add_argument('--sample_fps', type=int, default=16,
                        help='Frames per second for sampling (default: 16)')
    parser.add_argument('--sample_shift', type=float, default=5.0,
                        help='Shift value for sampling (default: 5.0)')
    parser.add_argument('--sample_guide_scale', type=float, default=5.0,
                        help='Guidance scale for sampling (default: 5.0)')
    parser.add_argument('--sample_steps', type=int, default=50,
                        help='Number of sampling steps (default: 50)')
    parser.add_argument('--sample_neg_prompt', type=str,
                        default='色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，水印，文字',
                        help='Negative prompt string (default: detailed Chinese negative prompt)')
    parser.add_argument('--base_seed', type=int, default=2025,
                        help='Base random seed (default: 2025)')
    parser.add_argument("--echoshot_model", type=str,
                        help='Path to the echoshot model')
    parser.add_argument("--wan_model", type=str,
                        help='Path to the wan model directory')
    parser.add_argument("--out_dir", type=str, default='outputs/',
                        help='Path to save the generated video')

    args = parser.parse_args()

    # args.checkpoint_path = os.path.join(args.model, 'EchoShot.pth')
    args.checkpoint_path = args.echoshot_model
    args.t5_checkpoint =  os.path.join(args.wan_model, 'models_t5_umt5-xxl-enc-bf16.pth')
    args.t5_tokenizer =  os.path.join(args.wan_model, 'google/umt5-xxl/')
    args.vae_checkpoint = os.path.join(args.wan_model, 'Wan2.1_VAE.pth')

    gen(args)

    