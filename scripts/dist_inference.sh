export CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7
torchrun --nproc_per_node=7 --nnodes=1 --master_addr="localhost" --master_port=1234 inference.py --vis
