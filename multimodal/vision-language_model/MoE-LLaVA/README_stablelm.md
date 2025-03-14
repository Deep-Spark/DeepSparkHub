# MoE-LLaVA-stablelm-2-1_6b
## Model description

MoE-LLaVA: Mixture of Experts for Large Vision-Language Models, the Language Models is stablelm-2-1_6b

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
[权重-stablelm-2-1_6b](http://files.deepspark.org.cn:880/deepspark/stablelm-2-1_6b)
格式如下:
```bash
stablelm-2-1_6b/
├── config.json
├── configuration_stablelm.py
├── generation_config.json
├── gitattributes
├── LICENSE.md
├── merges.txt
├── modeling_stablelm.py
├── model.safetensors
├── README.md
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
└── vocab.json
```


## Train
```bash
cd scripts/v1/stablelm-2-1_6b
bash pretrain.sh
```
