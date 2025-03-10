# WaveGlow

## Model Description

In our recent paper, we propose WaveGlow: a flow-based network capable of generating high quality speech from
mel-spectrograms. WaveGlow combines insights from Glow and WaveNet in order to provide fast, efficient and high-quality
audio synthesis, without the need for auto-regression. WaveGlow is implemented using only a single network, trained
using only a single cost function: maximizing the likelihood of the training data, which makes the training procedure
simple and stable.

## Model Preparation

### Prepare Resources

Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/) in the current directory.

- wget -c <https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2>;
- tar -jxvf LJSpeech-1.1.tar.bz2

### Install Dependencies

```sh
pip3 install -r requirements.txt 
```

## Model Training

```sh
# On single GPU
python3 train.py -c config.json

# Multiple GPUs on one machine
## Warning: DDP & AMP(mixed precision training set `"fp16_run": true` on `config.json`)
python3 distributed.py -c config.json
```

## Model Results

| Card Type | Prec. | Single Card | 8 Cards  |
|-----------|-------|------------:|:--------:|
| BI-V100   | FP32  |     196.845 | 1440.233 |
| BI-V100   | AMP   |     351.040 | 2400.745 |

## References

- [waveglow](https://github.com/NVIDIA/waveglow)
