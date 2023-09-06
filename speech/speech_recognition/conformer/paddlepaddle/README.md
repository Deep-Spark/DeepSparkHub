# Conformer

## Model description
Recently Transformer and Convolution neural network (CNN) based models have shown promising results in Automatic Speech Recognition (ASR), outperforming Recurrent neural networks (RNNs). Transformer models are good at capturing content-based global interactions, while CNNs exploit local features effectively. In this work, we achieve the best of both worlds by studying how to combine convolution neural networks and transformers to model both local and global dependencies of an audio sequence in a parameter-efficient way. To this regard, we propose the convolution-augmented transformer for speech recognition, named Conformer. Conformer significantly outperforms the previous Transformer and CNN based models achieving state-of-the-art accuracies. On the widely used LibriSpeech benchmark, our model achieves WER of 2.1%/4.3% without using a language model and 1.9%/3.9% with an external language model on test/testother. We also observe competitive performance of 2.7%/6.3% with a small model of only 10M parameters.

## Step 1:Installation
```bash
git clone --recursive -b r1.4 https://github.com/PaddlePaddle/PaddleSpeech.git
cd PaddleSpeech
pip3 install .
```

## Step 2:Preparing datasets
```bash
cd examples/aishell/asr1/
bash run.sh --stage 0 --stop_stage 0
```
run.sh will download and process the datasets, The download process may be slow, you can download the data_aishell.tgz from [wenet](http://openslr.magicdatatech.com/resources/33/data_aishell.tgz) and put it in the /path/to/PaddleSpeech/dataset/aishell/, then return to execute the above command.

## Step 3:Training
```bash
bash run.sh --stage 1 --stop_stage 3
```

## Results
|    GPUs     | IPS  |  CER   |
| :---------: | :--: | :----: |
| BI-V100 x 4 | 48.5 | 0.0495(checkpoint 81) |

