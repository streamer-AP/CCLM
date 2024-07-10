export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
torchrun --nproc_per_node=6 --nnodes=1 --master_addr="localhost" --master_port=1234 inference.py --mode test --vis