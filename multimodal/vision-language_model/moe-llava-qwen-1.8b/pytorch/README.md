
# MoE-LLaVA-Qwen-1.8B

## Model Description

MoE-LLaVA is a cutting-edge vision-language model that combines the Mixture of Experts (MoE) architecture with the
phi-2.7b language model. It excels in multimodal tasks by efficiently processing and integrating visual and textual
information. The model leverages expert networks to specialize in different aspects of vision-language understanding,
enabling more accurate and context-aware responses. MoE-LLaVA is particularly effective in applications requiring
complex reasoning across visual and linguistic domains, such as image captioning and visual question answering.

## Model Preparation

### Prepare Resources

Go to MoE-LLaVA toolbox.

```bash
cd <deepsparkhub_root>/toolbox/MoE-LLaVA
```

Dataset and weights need to link to current path in "MoE-LLaVA/"

Download from the [file server](http://files.deepspark.org.cn:880/deepspark)

The dataset path is as follows:

```bash
MoE-LLaVA/
├── gitattributes
├── llava_image
├── llava_image.zip
├── mimicit_tune
├── README.md
└── train_json
```

Get [clip-vit-large-patch14-336](http://files.deepspark.org.cn:880/deepspark/openai/).

The weights path is as follows:

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

Get [Qwen-1_8B](http://files.deepspark.org.cn:880/deepspark/Qwen-1_8B)

The weights path is as follows:

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

### Install Dependencies

```bash
cd MoE-LLaVA
pip install --upgrade pip  # enable PEP 660 support
pip3 install -e .
pip3 install --upgrade pydantic
```

## Model Training

```bash
cd scripts/v1/qwen
bash pretrain.sh
```

## References

- [MoE-LLaVA](https://github.com/PKU-YuanGroup/MoE-LLaVA)
