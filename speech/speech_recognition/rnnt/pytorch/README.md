# RNN-T

## Model description

Many machine learning tasks can be expressed as the transformation---or \emph{transduction}---of input sequences into output sequences: speech recognition, machine translation, protein secondary structure prediction and text-to-speech to name but a few. One of the key challenges in sequence transduction is learning to represent both the input and output sequences in a way that is invariant to sequential distortions such as shrinking, stretching and translating. Recurrent neural networks (RNNs) are a powerful sequence learning architecture that has proven capable of learning such representations. However RNNs traditionally require a pre-defined alignment between the input and output sequences to perform transduction. This is a severe limitation since \emph{finding} the alignment is the most difficult aspect of many sequence transduction problems. Indeed, even determining the length of the output sequence is often challenging. This paper introduces an end-to-end, probabilistic sequence transduction system, based entirely on RNNs, that is in principle able to transform any input sequence into any finite, discrete output sequence. Experimental results for phoneme recognition are provided on the TIMIT speech corpus.

## Step 1: Installing packages
Install required libraries:
```
bash install.sh
```

## Step 2: Preparing datasets
download LibriSpeech [http://www.openslr.org/12](http://www.openslr.org/12)
```
bash scripts/download_librispeech.sh $DATASET_DIR
```
preprocess LibriSpeech
```
bash scripts/preprocess_librispeech.sh $DATASET_DIR
```

## Step 3: Training

### Multiple GPUs on one machine

```
cd scripts
bash train_rnnt_1x8.sh $OUTPUT_DIR $DATA_DIR
```

Following conditions were tested, you can run any of them below:

|             | FP32                |
| ----------- | ------------------- |
| single card | `train_rnnt_1x1.sh` |  
| 4 cards     | `train_rnnt_1x4.sh` |
| 8 cards     | `train_rnnt_1x8.sh` |


## Results on BI-V100

| GPUs | FP16  | FPS | WER |
|------|-------|-----| ---------- |
| 1x8  | False | 20  | 0.058       |


## Reference
https://github.com/mlcommons/training/tree/master/rnn_speech_recognition/pytorch