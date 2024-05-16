# Llama2-7B RLHF

In this example, we use [Llama2-7b](https://huggingface.co/meta-llama/Llama-2-7b) and [Tiny-llama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-240k-503b) to do RLHF training. You can get them in huggingface through links provided.

**Notion: You would better to fine-tune this two models, then do RLHF training as below. So that can get good training result.**

## Step 1: Install

```
bash build_megatron-deepspeed.sh && bash install_megatron-deepspeed.sh
```

## Step 2: Dataset

Download dataset and convert it.

```
cd dataset && bash download_and_convert_dataset.sh
```

## Step 3: Checkpoint

Download checkpoints as above and put them to proper path (llama2_7b -> checkpoints/llama2-7b,  tinyllama_1.1b -> checkpoints/TinyLlama-1.1B), then convert checkpoints.

```
cd checkpoints && bash convert_hf_2_meg.sh
```

## Step 4: Train

```
cd examples/llama2
bash run_llama2_7b_rlhf_node1.sh
```
