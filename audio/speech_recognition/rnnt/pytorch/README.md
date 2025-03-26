# RNN-T

## Model Description

RNN-T (Recurrent Neural Network Transducer) is an end-to-end deep learning model designed for sequence-to-sequence
tasks, particularly in speech recognition. It combines RNNs with a transducer architecture to directly map input
sequences (like audio) to output sequences (like text) without requiring explicit alignment. The model consists of an
encoder network that processes the input, a prediction network that models the output sequence, and a joint network that
combines these representations. RNN-T handles variable-length input/output sequences and learns alignments automatically
during training. It's particularly effective for speech recognition as it can process continuous audio streams and
output text in real-time, achieving state-of-the-art performance on various benchmarks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 2.2.0     |  22.09  |

## Model Preparation

### Prepare Resources

```sh
# Download LibriSpeech [http://www.openslr.org/12](http://www.openslr.org/12)
bash scripts/download_librispeech.sh ${DATA_ROOT_DIR}

# Preprocess LibriSpeech

```sh
bash scripts/preprocess_librispeech.sh ${DATA_ROOT_DIR}
```

### Install Dependencies

```sh
bash install.sh
```

## Model Training

```sh
# Setup config yaml
sed -i "s#MODIFY_DATASET_DIR#${DATA_ROOT_DIR}/LibriSpeech#g" configs/baseline_v3-1023sp.yaml
```

### Multiple GPUs on one machine

```sh
mkdir -p output/
bash scripts/train_rnnt_1x8.sh output/ ${DATA_ROOT_DIR}/LibriSpeech
```

Following conditions were tested, you can run any of them below:

|             | FP32                |
|-------------|---------------------|
| single card | `train_rnnt_1x1.sh` |
| 4 cards     | `train_rnnt_1x4.sh` |
| 8 cards     | `train_rnnt_1x8.sh` |

## Model Results

| Model | GPUs       | FP16  | FPS | WER   |
|-------|------------|-------|-----|-------|
| RNN-T | BI-V100 x8 | False | 20  | 0.058 |

## References

- [mlcommons](https://github.com/mlcommons/training/tree/master/rnn_speech_recognition/pytorch)
