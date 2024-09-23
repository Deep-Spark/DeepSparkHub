# RLHF

## Step 1: Install

```
bash build_megatron-deepspeed.sh && bash install_megatron-deepspeed.sh
```

## Step 2: Dataset

Download dataset and convert it.

```
cd dataset && bash convert_dahoas.sh
```

## Step 3: Checkpoint

Download and convert checkpoints.

```
cd checkpoints && bash download_rlhf_checkpoints.sh
bash convert_hf_2_meg.sh
```

## Step 4: Train

```
cd examples/llama2
bash run_llama2_7b_rlhf_node1.sh
```
