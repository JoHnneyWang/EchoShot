import os
import os.path as osp
import torch
from easydict import EasyDict
import torch
from datetime import datetime
from torch.distributed.fsdp import ShardingStrategy



#------------------------ environment ------------------------#
cfg = EasyDict(__name__='Config: Wanx I2V')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
os.environ['PYTHONHASHSEED'] = '2024'  # ensure consistent output from `hash()`
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'



#------------------------ model ------------------------#s
# t5
cfg.t5_model = 'umt5_xxl'
cfg.t5_dtype = torch.bfloat16
cfg.text_len = 512
cfg.t5_checkpoint = 'Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth'
cfg.t5_tokenizer = 'Wan2.1-T2V-1.3B/google/umt5-xxl/'

# vae
cfg.vae_checkpoint = 'Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'
cfg.downsample = (4, 16, 16)

# transformer-1.3B
cfg.patch_size = (1, 2, 2)
cfg.dim = 1536
cfg.ffn_dim = 8960
cfg.freq_dim = 256
cfg.num_heads = 12
cfg.num_layers = 30
cfg.window_size = (-1, -1)
cfg.qk_norm = True
cfg.cross_attn_norm = True
cfg.eps = 1e-6



# ----------------- inference ----------------------------
H = 480
W = 832
T = 125

t = (T-1)//4+1
cfg.param_dtype = torch.bfloat16
cfg.num_train_timesteps = 1000

cfg.sample_fps = 16
cfg.sample_shift = 5.0
cfg.sample_guide_scale = 5.0
cfg.sample_steps = 50
cfg.sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，水印，文字'
cfg.base_seed = 2025
cfg.max_seq_len = t * W * H // 16 // 16
cfg.target_shape = (16, t, H//8, W//8) # video size // 8
