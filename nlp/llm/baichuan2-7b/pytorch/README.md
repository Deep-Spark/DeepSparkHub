# Baichuan2-7B (DeepSpeed)

## Model Description

Baichuan2-7B is an advanced open-source large language model developed by Baichuan Intelligent Technology. Trained on a
high-quality corpus of 2.6 trillion tokens, it excels in both Chinese and English language tasks. The model demonstrates
superior performance across various benchmarks, offering capabilities in text generation, comprehension, and
domain-specific applications. Baichuan2-7B's architecture is optimized for efficiency and scalability, making it
suitable for both academic research and commercial use. Its open-source nature encourages innovation and development in
the field of natural language processing.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V150 | 3.4.0     |  24.06  |

## Model Preparation

### Prepare Resources

Load model weight, and fix configuration_baichuan.py.
  
```sh
pip install modelscope
cd fine-tune/
python3 ./get_Baichuan2_model.py
mkdir -p /home/model_zoo/nlp
mv /root/.cache/modelscope/hub/baichuan-inc/Baichuan2-7B-Base /home/model_zoo/nlp
mv configuration_baichuan.py /home/model_zoo/nlp/Baichuan2-7B-Base

# get belle_chat_ramdon_10k data
mkdir -p data/
wget -c --no-check-certificate https://raw.githubusercontent.com/baichuan-inc/Baichuan2/main/fine-tune/data/belle_chat_ramdon_10k.json -P data/
```

### Install Dependencies

```sh
cd ../
pip install -r requirements.txt

cd fine-tune/
pip install -r requirements.txt
```

## Model Training

Fine-tuning.

```sh
bash ./run_sft.sh
```

## Model Results

| Model        | GPUs       | Epochs | train_samples_per_second |
|--------------|------------|--------|--------------------------|
| Baichuan2-7B | BI-V150 x8 | 1      | 10.674                   |

## References

- [Baichuan2](https://github.com/baichuan-inc/Baichuan2)
