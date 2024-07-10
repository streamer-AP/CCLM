export CUDA_VISIBLE_DEVICES=7
torchrun --nproc_per_node=1 --nnodes=1 --master_addr="localhost" --master_port=1235  inference_loc_dist_test.py
```