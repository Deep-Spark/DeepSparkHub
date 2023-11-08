# Megatron-DeepSpeed LLaMA2-7B

## Model description

LLaMA2 is a large language model released by Meta in 2023, with parameters ranging from 7B to 70B. Compared to LLaMA, the training corpus of LLaMA2 is 40% longer, and the context length has been upgraded from 2048 to 4096, allowing for understanding and generating longer texts. 

## Step 1: Installation

```
bash build_megatron-deepspeed.sh && bash install_megatron-deepspeed.sh
pip3 install urllib3==1.23
```

## Step 2: Preparing datasets

```
cd dataset
mkdir BookCorpusDataset && cd BookCorpusDataset
wget https://the-eye.eu/public/AI/pile_neox/data/BookCorpusDataset_text_document.bin
wget https://the-eye.eu/public/AI/pile_neox/data/BookCorpusDataset_text_document.idx
```

## Step 3: Training

```
export NCCL_SOCKET_IFNAME="eth0"
bash run_llama2_7b_1node.sh
```

If the torchrun command cannot be found，you can execute:

```
ln -s /usr/local/corex-3.1.0/lib64/python3/dist-packages/bin/torchrun /usr/local/bin/
```

## Results
