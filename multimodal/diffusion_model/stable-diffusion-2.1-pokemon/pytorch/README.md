## 安装环境
### 安装 transformers
```bash
git clone ssh://git@bitbucket.iluvatar.ai:7999/apptp/transformers.git
cd transformers
python3 setup.py install
```
### 安装diffusers
```bash
cd diffusers
pip3 install pip3 install -r examples/text_to_image/requirements.txt
bash build_diffusers.sh && bash install_diffusers.sh
```
_默认已经安装的包有：torchvision,ixformer,flash-attn,deepspeed,apex_
_上述包最好使用较新的daily，不然可能会有功能不支持_

## 下载数据
```bash
mkdir -p pokemon-blip-captions 
download here: http://10.150.9.95/swapp/datasets/multimodal/stable_diffusion/pokemon-blip-captions
wget http://10.150.9.95/swapp/datasets/multimodal/stable_diffusion/stabilityai.tar   # sd2.1 权重
tar -xvf stabilityai.tar
```

*sdxl权重链接：http://sw.iluvatar.ai/download/apps/datasets/aigc/xl/stable-diffusion-xl-base-1.0.tar.gz*

*sd1.5权重链接：http://10.150.9.95/swapp/pretrained/multimodal/stable-diffusion/stable-diffusion-v1-5.zip*


## 训练
*以下脚本中包含的数据和预训练权重位置需要根据实际存放位置调整*
### sd2.1 训练
```bash
$ bash run_sd_2.1.sh   # 多卡
$ bash run_sd_2.1_single.sh   # 单卡
```
### sd1.5 训练
```bash
$ bash run_sd_1.5.sh   # 多卡
$ bash run_sd_1.5_single.sh   # 单卡
```
### sdxl 训练
```bash
$ bash run_sd_xl.sh   # 多卡
```
