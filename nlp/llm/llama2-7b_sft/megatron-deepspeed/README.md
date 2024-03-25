# Megatron-DeepSpeed Llama-2-7B SFT

## Model description

Llama 2 is a large language model released by Meta in 2023, with parameters ranging from 7B to 70B. Compared to LLaMA, the training corpus of Llama 2 is 40% longer, and the context length has been upgraded from 2048 to 4096, allowing for understanding and generating longer texts.

## Step 1: Installation

```bash
# Install sqlite3
wget https://sqlite.org/2019/sqlite-autoconf-3290000.tar.gz
tar zxvf sqlite-autoconf-3290000.tar.gz
pushd sqlite-autoconf-3290000
./configure
make && make install
popd

# Reinstall Python
wget https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tar.xz
tar xvf Python-3.7.9.tar.xz
pushd Python-3.7.9
./configure LDFLAGS="-L/usr/local/lib" CPPFLAGS="-I/usr/local/include" --prefix=/usr/bin
make && make install

cp /usr/bin/lib/python3.7/lib-dynload/_sqlite3.cpython-37m-x86_64-linux-gnu.so /usr/local/lib/python3.7/lib-dynload/_sqlite3.so
popd

# Install Megatron-Deepspeed
bash build_megatron-deepspeed.sh && bash install_megatron-deepspeed.sh
```

## Step 2: Preparing datasets

```bash
cd dataset/
bash download_and_convert_dataset.sh
```

## Step 3: Download and convert HF weight

You can download huggingface llama2-7b pretrained model from [here](https://huggingface.co/meta-llama/Llama-2-7b), and use below script to convert it.

```bash
cd checkpoints
bash convert_hf_2_meg.sh
```

## Step 4: Training

```bash
cd examples/llama2
bash run_meg_llama2_7b_sft.sh
```

If the torchrun command cannot be found，you can execute:

```
ln -s /usr/local/corex-3.1.0/lib64/python3/dist-packages/bin/torchrun /usr/local/bin/
```

## Results
| GPUs       | Toolbox   | Model       | Training speed   |
|:-----------:|:---------:|:----------:|:----------------:|
| BI-V100 x8 | Megatron-DeepSpeed | LLaMA2-7B SFT|1.146 samples/sec |

## Reference
- [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
