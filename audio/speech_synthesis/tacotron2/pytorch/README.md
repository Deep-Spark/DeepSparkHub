# Tacotron2

## Model Description

Tacotron2 is an end-to-end neural text-to-speech synthesis system that directly converts text into natural-sounding
speech. It combines a sequence-to-sequence network that generates mel-spectrograms from text with a WaveNet-based
vocoder to produce high-quality audio. The model achieves near-human speech quality with a Mean Opinion Score (MOS) of
4.53, rivaling professional recordings. Its architecture simplifies traditional speech synthesis pipelines by using
learned acoustic representations, enabling more natural prosody and articulation while maintaining computational
efficiency.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 2.2.0     |  22.09  |

## Model Preparation

### Prepare Resources

1.Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/) in the current directory;

- wget -c <https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2>;
- tar -jxvf LJSpeech-1.1.tar.bz2;

### Install Dependencies

```sh
pip3 install -r requirements.txt 
```

## Model Training

First, create a directory to save output and logs.

```sh
mkdir outdir logdir
```

##

```sh
# On single GPU
python3 train.py --output_directory=outdir --log_directory=logdir --target_val_loss=0.5
 
# Multiple GPUs on one machine
python3 -m multiproc train.py --output_directory=outdir --log_directory=logdir --hparams=distributed_run=True --target_val_loss=0.5

# Multiple GPUs on one machine (AMP)
python3 -m multiproc train.py --output_directory=outdir --log_directory=logdir --hparams=distributed_run=True,fp16_run=True --target_val_loss=0.5
```

## Model Results

| GPU        | FP16 | FPS | Score(MOS) |
|------------|------|-----|------------|
| BI-V100 x8 | True | 9.2 | 4.460      |

| Convergence criteria | Configuration (x denotes number of GPUs) | Performance | Accuracy | Power（W） | Scalability | Memory utilization（G） | Stability |
|----------------------|------------------------------------------|-------------|----------|------------|-------------|-------------------------|-----------|
| score(MOS):4.460     | SDK V2.2,bs:128,8x,AMP                   | 77          | 4.46     | 128\*8     | 0.96        | 18.4\*8                 | 1         |

## References

- [tacotron2](https://github.com/NVIDIA/tacotron2)
