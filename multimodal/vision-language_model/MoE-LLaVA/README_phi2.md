# MoE-LLaVA-phi-2.7b
## Model description

MoE-LLaVA: Mixture of Experts for Large Vision-Language Models, the Language Models is phi-2.7b


## Prepare

### Install requirements

```bash

cd MoE-LLaVA
pip install --upgrade pip  # enable PEP 660 support
pip3 install -e .
pip3 install --upgrade pydantic

```
### load data and weights
数据集和权重需要链接到当前目录 MoE-LLaVA 里
[数据集地址]（http://files.deepspark.org.cn:880/deepspark/）
格式如下:
```bash
MoE-LLaVA/
├── gitattributes
├── llava_image
├── llava_image.zip
├── mimicit_tune
├── README.md
└── train_json
```
[权重-clip-vit-large-patch14-336](http://files.deepspark.org.cn:880/deepspark/openai/)
格式如下:
```bash
openai/
└── clip-vit-large-patch14-336
    ├── config.json
    ├── configuration.json
    ├── merges.txt
    ├── preprocessor_config.json
    ├── pytorch_model.bin
    ├── README.md
    ├── special_tokens_map.json
    ├── tf_model.h5
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── vocab.json
```
[权重-phi-2.7b](http://files.deepspark.org.cn:880/deepspark/phi-2)
格式如下:
```bash
phi-2/
├── added_tokens.json
├── CODE_OF_CONDUCT.md
├── config.json
├── generation_config.json
├── gitattributes
├── LICENSE
├── merges.txt
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
├── NOTICE.md
├── README.md
├── SECURITY.md
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
└── vocab.json

```


## Train
```bash
cd scripts/v1/phi2
bash pretrain.sh
```
