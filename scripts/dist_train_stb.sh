export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
torchrun --nproc_per_node=6 --nnodes=1  --master_port 12350  train_counter_stb.py
