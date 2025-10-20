# Conformer

## Model Description

Recently Transformer and Convolution neural network (CNN) based models have shown promising results in Automatic Speech
Recognition (ASR), outperforming Recurrent neural networks (RNNs). Transformer models are good at capturing
content-based global interactions, while CNNs exploit local features effectively. In this work, we achieve the best of
both worlds by studying how to combine convolution neural networks and transformers to model both local and global
dependencies of an audio sequence in a parameter-efficient way. To this regard, we propose the convolution-augmented
transformer for speech recognition, named Conformer. Conformer significantly outperforms the previous Transformer and
CNN based models achieving state-of-the-art accuracies. On the widely used LibriSpeech benchmark, our model achieves WER
of 2.1%/4.3% without using a language model and 1.9%/3.9% with an external language model on test/testother. We also
observe competitive performance of 2.7%/6.3% with a small model of only 10M parameters.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| BI-V150 | 4.3.0     |  25.12  |

## Model Preparation

### Prepare Resources
```bash
mkdir -p data/datasets/LibriSpeech
cd data/datasets/LibriSpeech
wget http://files.deepspark.org.cn:880/deepspark/conformer/LibriSpeech/dev-clean.tar.gz
wget http://files.deepspark.org.cn:880/deepspark/conformer/LibriSpeech/train-clean-100.tar.gz
tar -xf dev-clean.tar.gz
tar -xf train-clean-100.tar.gz
mv LibriSpeech/* ./

└── data/datasets/LibriSpeech
    ├── train-clean-100.tar.gz
    ├── dev-clean.tar.gz
    ├── dev-clean
    └── train-clean-100
    └── ...

mkdir -p data/model_zoo/sentencepieces
cd data/model_zoo/sentencepieces
wget http://files.deepspark.org.cn:880/deepspark/conformer/sentencepieces/sp.model
wget http://files.deepspark.org.cn:880/deepspark/conformer/sentencepieces/sp.vocab
```

### Install Dependencies

```sh
apt install -y numactl libsndfile1
pip3 install http://files.deepspark.org.cn:880/deepspark/conformer/IXPyLogger-1.0.0-py3-none-any.whl
pip3 install numpy==1.26.4
pip3 install -r requirements.txt

wget https://librosa.org/data/audio/admiralbob77_-_Choice_-_Drum-bass.ogg
mkdir -p ~/.cache/librosa/
mv admiralbob77_-_Choice_-_Drum-bass.ogg ~/.cache/librosa/
```

## Model Training

```bash
bash run_training.sh --data_dir=./data \
    --max_steps=800 \
    --quality_target=1.6 \
    --batch_size=8 \
    --eval_freq=400 \
    --ddp \
    --max_steps=800 \
    --quality_target=1.6 \
    --eval_freq=400
```

## Model Results
| GPU         | tps  | wer                   |
|-------------|------|-----------------------|
| BI-V150 x 8 | 127.7341 | 1.4652 |
