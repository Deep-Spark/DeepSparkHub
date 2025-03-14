
# MoE-LLaVA-Qwen-1_8B
## Model description

MoE-LLaVA: Mixture of Experts for Large Vision-Language Models, the Language Models is Qwen-1_8B


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
[权重-Qwen-1_8B](http://files.deepspark.org.cn:880/deepspark/Qwen-1_8B)
格式如下:
```bash
Qwen-1_8B/
├── assets
│   ├── logo.jpg
│   ├── qwen_tokenizer.png
│   ├── tokenizer.png
│   └── wechat.png
├── cache_autogptq_cuda_256.cpp
├── cache_autogptq_cuda_kernel_256.cu
├── config.json
├── configuration_qwen.py
├── cpp_kernels.py
├── generation_config.json
├── gitattributes
├── LICENSE
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── modeling_qwen.py
├── model.safetensors.index.json
├── NOTICE
├── qwen_generation_utils.py
├── qwen.tiktoken
├── README.md
├── tokenization_qwen.py
└── tokenizer_config.json
```


## Train
```bash
cd scripts/v1/qwen
bash pretrain.sh
```



