export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 --nnodes=1  --master_port 12350  train_counter.py
