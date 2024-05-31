# Baichuan2-7B

## Model description

Baichuan 2 is the new generation of open-source large language models launched by Baichuan Intelligent Technology. It was trained on a high-quality corpus with 2.6 trillion tokens.  Baichuan 2 achieved the best performance of its size on multiple authoritative Chinese, English, and multi-language general and domain-specific benchmarks. All versions are fully open to academic research. Developers only need to apply via email and obtain official commercial permission to use it for free commercially.

## Step 1: Preparing datasets

Load model weight, and fix configuration_baichuan.py
  
```bash
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

## Step 2: Installation

```bash
cd ../
pip install -r requirements.txt

cd fine-tune/
pip install -r requirements.txt
```

## Step 3: Training

Fine-tuning

```bash
bash ./run_sft.sh
```

## Results

| GPUs       | Epochs | train_samples_per_second |
|------------|--------|-----|
| BI-V150 x8 | 1      | 10.674 |

## Reference

- [Baichuan2] <https://github.com/baichuan-inc/Baichuan2>
