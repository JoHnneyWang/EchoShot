export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nnodes ${WORLD_SIZE} \
    --nproc_per_node 8 \
    --node_rank=${RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    train.py