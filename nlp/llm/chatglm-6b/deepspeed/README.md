# DeepSpeed ChatGLM-6B

## Model description
ChatGLM-6B is an open bilingual language model based on [General Language Model (GLM)](https://github.com/THUDM/GLM) framework, with 6.2 billion parameters. With the quantization technique, users can deploy locally on consumer-grade graphics cards (only 6GB of GPU memory is required at the INT4 quantization level). 

ChatGLM-6B uses technology similar to ChatGPT, optimized for Chinese QA and dialogue. The model is trained for about 1T tokens of Chinese and English corpus, supplemented by supervised fine-tuning, feedback bootstrap, and reinforcement learning wit human feedback. With only about 6.2 billion parameters, the model is able to generate answers that are in line with human preference.

## Step 1: Installation
```shell
# Install requirements
pip3 install -r requirements.txt

# Install sqlite
yum install sqlite-devel
wget https://www.sqlite.org/2018/sqlite-autoconf-3240000.tar.gz
tar -xvzf sqlite-autoconf-3240000.tar.gz
cd sqlite-autoconf-3240000/
./configure --prefix=/usr/local/sqlite
make -j4 && make install

# Install Python 3.7.9
wget https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tgz
tar -xf Python-3.7.9.tgz ; cd Python-3.7.9
./configure --enable-loadable-sqlite-extensions
make && make install
```

### Install DeepSpeed
ChatGLM-6B model is using DeepSpeed toolbox. Before you run this model, you need to install DeepSpeed first.
```shell
cd ../../../../../toolbox/DeepSpeed/
yum install libaio libaio-devel -y
bash clean_deepspeed.sh
bash build_deepspeed.sh
bash install_deepspeed.sh
cd -
```

## Step 2: Preparing datasets

ADGEN is a large-scale dataset for advertisement text generation proposed by researchers from Hong Kong University of Science and Technology in 2018.
Go to [Google Drive](https://drive.google.com/file/d/13_vf0xRTQsyneRKdD1bZIr93vBGOczrk/view?usp=sharing) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1), download the processed ADGEN dataset, and decompress AdvertiseGen directory.

If you want to load the model locally, you can download the model implementation ( `13GB` ) from [Hugging Face Hub](https://huggingface.co/THUDM/chatglm-6b) 
```shell
# Install lfs
yum install -y rh-git218-git-lfs.x86_64
source /opt/rh/rh-git218/enable
# Get huggingface dataset
git lfs install
git config --global http.sslVerify false
git clone https://huggingface.co/THUDM/chatglm-6b
```

## Step 3: Training
If you load the model locally, you can change `THUDM/chatglm-6b` in `ds_train_finetune.sh` to your local model path.

```shell
cd ptuning/
bash ds_train_finetune.sh
```
## Results
| GPUs       | Toolbox   | Model       | Training speed   |
|:-----------:|:---------:|:----------:|:----------------:|
| BI-V100 x8 | DeepSpeed | ChatGLM-6B |0.995 samples/sec |

## Reference
[THUDM/ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)