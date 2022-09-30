# Tacotron2 

## Model description

This paper describes Tacotron 2, a neural network architecture for speech synthesis directly from text. The system is composed of a recurrent sequence-to-sequence feature prediction network that maps character embeddings to mel-scale spectrograms, followed by a modified WaveNet model acting as a vocoder to synthesize timedomain waveforms from those spectrograms. Our model achieves a mean opinion score (MOS) of 4.53 comparable to a MOS of 4.58 for professionally recorded speech. To validate our design choices, we present ablation studies of key components of our system and evaluate the impact of using mel spectrograms as the input to WaveNet instead of linguistic, duration, and F_0 features. We further demonstrate that using a compact acoustic intermediate representation enables significant simplification of the WaveNet architecture.

## Step 1: Installing packages
```  
pip3 install -r requirements.txt 
```

## Step 2: Preparing datasets
1.Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/) in the current directory;
  - wget -c https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2;
  - tar -jxvf LJSpeech-1.1.tar.bz2;


## Step 3: Training

First, create a directory to save output and logs.

```
$ mkdir outdir logdir
```

### On single GPU
```
$ python3 train.py --output_directory=outdir --log_directory=logdir --target_val_loss=0.5
```

### Multiple GPUs on one machine
```
$ python3 -m multiproc train.py --output_directory=outdir --log_directory=logdir --hparams=distributed_run=True --target_val_loss=0.5
```


### Multiple GPUs on one machine (AMP)
```
$ python3 -m multiproc train.py --output_directory=outdir --log_directory=logdir --hparams=distributed_run=True,fp16_run=True --target_val_loss=0.5
```

## Results on BI-V100

| GPUs | FP16 | FPS | Score(MOS) |
|------| ---- |-----| ---------- |
| 1x8  | True | 9.2 | 4.460      |

| Convergence criteria | Configuration (x denotes number of GPUs) | Performance | Accuracy | Power（W） | Scalability | Memory utilization（G） | Stability |
|----------------------|------------------------------------------|-------------|----------|------------|-------------|-------------------------|-----------|
| score(MOS):4.460     | SDK V2.2,bs:128,8x,AMP                   | 77          | 4.46     | 128\*8     | 0.96        | 18.4\*8                 | 1         |


## Reference
https://github.com/NVIDIA/tacotron2