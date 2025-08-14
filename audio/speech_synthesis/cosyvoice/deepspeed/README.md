# CosyVoice2

## Model Description

CosyVoice2-0.5B is a small speech model designed to understand and generate human-like speech. It can be used for tasks like voice assistants, text-to-speech, or voice cloning. With 0.5 billion parameters, it is lightweight and works well on devices with limited computing power. It focuses on natural-sounding voices and easy customization.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| BI-V150 | 4.3.0     |  25.09  |

## Model Preparation

### Prepare Resources

Pretrained model: <https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B>

Dataset: <https://openslr.elda.org/resources/60/>

```bash
mkdir -p /root/datasets/openslr/libritts
cd /root/datasets/openslr/libritts
wget https://openslr.elda.org/resources/60/train-clean-100.tar.gz
wget https://openslr.elda.org/resources/60/dev-clean.tar.gz
tar -xvzf train-clean-100.tar.gz
tar -xvzf dev-clean.tar.gz
```

### Install Dependencies

Contact the Iluvatar administrator to get the missing packages:

- transformers-4.45.2+corex.4.3.0-py3-none-any.whl
- deepspeed-0.16.4+corex.4.3.0-cp310-cp310-linux_x86_64.whl

```bash
pip3 install -r requirements.txt
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
# If you failed to clone the submodule due to network failures, please run the following command until success
cd CosyVoice
git submodule update --init --recursive

mkdir -p pretrained_models
# download CosyVoice2-0.5B model into pretrained_models dir

# If you encounter sox compatibility issues
# ubuntu
sudo apt-get install sox libsox-dev
# centos
sudo yum install sox sox-devel
```

## Model Training

```bash
cp ../run_dpo.sh examples/libritts/cosyvoice2/
cp ../run.sh examples/libritts/cosyvoice2/
cd examples/libritts/cosyvoice2/

export PYTHONPATH=../../../:../../../third_party/Matcha-TTS:$PYTHONPATH
bash run.sh
```

## Model Results

## References

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice/commit/0a496c18f78ca993c63f6d880fcc60778bfc85c1)