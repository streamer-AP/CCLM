export CUDA_VISIBLE_DEVICES=6,7
torchrun --nproc_per_node=2 --nnodes=1  --master_port 12350  train_counter_stb.py
