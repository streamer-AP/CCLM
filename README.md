# Compactness Driven Co-learning for Crowd Counting and Localization

Official PyTorch implementation of **"Compactness Driven Co-learning for Crowd Counting and Localization"**.

## Installation

### 1. Create Conda Environment

```bash
conda create -n cclm python=3.10 cmake=3.16.3
conda activate cclm
```

### 2. Install Dependencies

```bash
pip install -r scripts/requirements.txt
```

## Dataset Preparation

Please follow [FIDTM](https://github.com/dk-liang/FIDTM) to process the datasets into COCO format.

The expected directory structure is:

```
data/
├── part_A_final/
│   ├── train_data/
│   ├── test_data/
│   ├── train_data_annotation.json
│   └── test_data_annotation.json
├── part_B_final/
├── NWPU/
└── FDST/
```

## Training

### 1. Modify Configuration

Edit `configs/fidt_ucl.json`:

- Change `Dataset.train.name` to the target dataset: `sta_crowd`, `stb_crowd`, `nwpu_crowd`, or `fdst_crowd`
- Update `ann_file` and `img_prefix` to your actual dataset paths

### 2. Launch Training

For single-node multi-GPU training (4 GPUs):

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 --nnodes=1 --master_port 12350 train_counter.py --config configs/fidt_ucl.json
```


## Evaluation

### 1. Compile NMS Module

Before evaluation, compile the Cython NMS module:

```bash
python setup.py build_ext --inplace
```

### 2. Run Evaluation

```bash
export CUDA_VISIBLE_DEVICES=0
python inference_nwpu.py --config configs/fidt_ucl.json --ckpt path/to/your/best.pth
```

Replace `path/to/your/best.pth` with the actual path to your trained checkpoint.


