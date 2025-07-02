export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun eval.py --ckpt '.pth' \
    --prompt 'prompts.json' \
    --seed 2025 \
    --postfix 'demo' \
    --out_dir 'outputs/demo'