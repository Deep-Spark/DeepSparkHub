# MoE-LLaVA-Phi2-2.7B

## Model Description

MoE-LLaVA is a cutting-edge vision-language model that combines the Mixture of Experts (MoE) architecture with the
phi-2.7b language model. It excels in multimodal tasks by efficiently processing and integrating visual and textual
information. The model leverages expert networks to specialize in different aspects of vision-language understanding,
enabling more accurate and context-aware responses. MoE-LLaVA is particularly effective in applications requiring
complex reasoning across visual and linguistic domains, such as image captioning and visual question answering.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |

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

Get [phi-2.7b](http://files.deepspark.org.cn:880/deepspark/phi-2)

The weights path is as follows:

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

### Install Dependencies

```bash
cd MoE-LLaVA
pip install --upgrade pip  # enable PEP 660 support
pip3 install -e .
pip3 install --upgrade pydantic
```

## Model Training

```bash
cd scripts/v1/phi2
bash pretrain.sh
```

## References

- [MoE-LLaVA](https://github.com/PKU-YuanGroup/MoE-LLaVA)
