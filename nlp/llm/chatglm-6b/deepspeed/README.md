# ChatGLM-6B

## Model description
ChatGLM-6B is an open bilingual language model based on [General Language Model (GLM)](https://github.com/THUDM/GLM) framework, with 6.2 billion parameters. With the quantization technique, users can deploy locally on consumer-grade graphics cards (only 6GB of GPU memory is required at the INT4 quantization level). 

ChatGLM-6B uses technology similar to ChatGPT, optimized for Chinese QA and dialogue. The model is trained for about 1T tokens of Chinese and English corpus, supplemented by supervised fine-tuning, feedback bootstrap, and reinforcement learning wit human feedback. With only about 6.2 billion parameters, the model is able to generate answers that are in line with human preference.

## Step 1: Installation
```shell
pip3 install -r requirements.txt

yum install sqlite-devel
wget https://www.sqlite.org/2018/sqlite-autoconf-3240000.tar.gz
tar -xvzf sqlite-autoconf-3240000.tar.gz
cd sqlite-autoconf-3240000/
./configure --prefix=/usr/local/sqlite
make -j4 && make install
 
wget https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tgz
tar -xf Python-3.7.9.tgz ; cd Python-3.7.9
./configure --enable-loadable-sqlite-extensions
make && make install

```

### Install DeepSpeed
ChatGlM model is using Deepspeed toolbox. Before you run this model, you need to setup Deepspeed first.
```shell
cd ../../../../../toolbox/DeepSpeed/
yum install libaio libaio-devel -y
bash clean_deepspeed.sh
bash build_deepspeed.sh
bash install_deepspeed.sh
cd -
```

## Step 2: Preparing datasets
ADGEN 数据集任务为根据输入（content）生成一段广告词（summary）。

```json
{
    "content": "类型#上衣*版型#宽松*版型#显瘦*图案#线条*衣样式#衬衫*衣袖型#泡泡袖*衣款式#抽绳",
    "summary": "这件衬衫的款式非常的宽松，利落的线条可以很好的隐藏身材上的小缺点，穿在身上有着很好的显瘦效果。领口装饰了一个可爱的抽绳，漂亮的绳结展现出了十足的个性，配合时尚的泡泡袖型，尽显女性甜美可爱的气息。"
}
```
From [Google Drive](https://drive.google.com/file/d/13_vf0xRTQsyneRKdD1bZIr93vBGOczrk/view?usp=sharing)  or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1) download the processed ADGEN dataset, and put the decompressed AdvertiseGen directory into this directory.

If you want to load the model locally, you can download the model implementation ( `13GB` ) from [Hugging Face Hub](https://huggingface.co/THUDM/chatglm-6b) 
```shell
# install lfs on centos
yum install -y rh-git218-git-lfs.x86_64
source /opt/rh/rh-git218/enable
# get huggingface dataset
git lfs install
git config --global http.sslVerify false
git clone https://huggingface.co/THUDM/chatglm-6b
```

## Step 3: Training
If you load the model locally, You can change `THUDM/chatglm-6b` in `ds_train_finetune.sh` to your local model path.

```shell
cd ptuning
bash ds_train_finetune.sh
```
## Results
| Model       | Toolbox   | GPUs       | Train speed      |
|:-----------:|:---------:|:----------:|:----------------:|
| ChatGLM-6B  | DeepSpeed | BI-V100 x8 |0.995 samples/sec |

## Reference
[Reference](https://github.com/THUDM/ChatGLM-6B)