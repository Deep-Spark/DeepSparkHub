# PP-TTS-FastSpeech2

## Model Description

Non-autoregressive text to speech (TTS) models such as FastSpeech can synthesize speech significantly faster than
previous autoregressive models with comparable quality. FastSpeech 2s is the first attempt to directly generate speech
waveform from text in parallel, enjoying the benefit of fully end-to-end inference. Experimental results show that 1)
FastSpeech 2 achieves a 3x training speed-up over FastSpeech, and FastSpeech 2s enjoys even faster inference speed; 2)
FastSpeech 2 and 2s outperform FastSpeech in voice quality, and FastSpeech 2 can even surpass autoregressive models.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.0     |  23.09  |

## Model Preparation

### Prepare Resources

### Download and Extract

Download CSMSC(BZNSYP) from this [Website.](https://aistudio.baidu.com/datasetdetail/36741) and extract it to
./datasets. Then the dataset is in the directory ./datasets/BZNSYP.

### Get MFA Result and Extract

We use [MFA](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) to get durations for fastspeech2. You can
download from here
[baker_alignment_tone.tar.gz](https://paddlespeech.bj.bcebos.com/MFA/BZNSYP/with_tone/baker_alignment_tone.tar.gz).

Put the data directory structure like this:

```sh
tts3
├── baker_alignment_tone
├── conf
├── datasets
│   └── BZNSYP
│       ├── PhoneLabeling
│       ├── ProsodyLabeling
│       └── Wave
├── local
└── ...
```

Change the rootdir of dataset in ./local/preprocess.sh to the dataset path. Like this: `--rootdir=./datasets/BZNSYP`

### Install Dependencies

```sh
# Pip the requirements
pip3 install -r requirements.txt

# Clone the repo
git clone https://github.com/PaddlePaddle/PaddleSpeech.git
cd PaddleSpeech/examples/csmsc/tts3
```

```sh
# Install sqlite3
wget https://sqlite.org/2019/sqlite-autoconf-3290000.tar.gz
tar zxvf sqlite-autoconf-3290000.tar.gz
cd sqlite-autoconf-3290000
./configure
make && make install
cd ..

wget https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tar.xz
tar xvf Python-3.7.9.tar.xz
cd Python-3.7.9
./configure LDFLAGS="-L/usr/local/lib" CPPFLAGS="-I/usr/local/include" --prefix=/usr/bin
make && make install

cp /usr/bin/lib/python3.7/lib-dynload/_sqlite3.cpython-37m-x86_64-linux-gnu.so /usr/local/lib/python3.7/lib-dynload/_sqlite3.so
```

```sh
# Update GCC lib
wget http://ftp.gnu.org/gnu/gcc/gcc-8.3.0/gcc-8.3.0.tar.gz
tar -zxvf gcc-8.3.0.tar.gz
yum -y install bzip2
cd gcc-8.3.0
./contrib/download_prerequisites
mkdir build
cd build/
../configure -enable-checking=release -enable-languages=c,c++ -disable-multilib
make -j 10
make install

cp /usr/local/lib64/libstdc++.so.6.0.25 /usr/lib64
cd /usr/lib64
rm -rf libstdc++.so.6
ln -s libstdc++.so.6.0.25 libstdc++.so.6
```

### Preprocess Data

```sh
PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning' ./run.sh --stage 0 --stop-stage 0
```

When it is done. A `dump` folder is created in the current directory. The structure of the dump folder is listed below.

```sh
dump
├── dev
│   ├── norm
│   └── raw
├── phone_id_map.txt
├── speaker_id_map.txt
├── test
│   ├── norm
│   └── raw
└── train
    ├── energy_stats.npy
    ├── norm
    ├── pitch_stats.npy
    ├── raw
    └── speech_stats.npy
```

### Model Training

You can choose use how many gpus for training by changing gups parameter in run.sh file and ngpu parameter in ./local/train.sh file.

```bash
PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning' ./run.sh --stage 1 --stop-stage 1
```

#### Synthesizing

We use [parallel wavegan](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/csmsc/voc1) as the neural
vocoder. Download pretrained parallel wavegan model from
[pwg_baker_ckpt_0.4.zip](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_baker_ckpt_0.4.zip) and
unzip it.

```sh
unzip pwg_baker_ckpt_0.4.zip
```

Parallel WaveGAN checkpoint contains files listed below.

```sh
pwg_baker_ckpt_0.4
├── pwg_default.yaml               # default config used to train parallel wavegan
├── pwg_snapshot_iter_400000.pdz   # model parameters of parallel wavegan
└── pwg_stats.npy                  # statistics used to normalize spectrogram when training parallel wavegan
```

Run synthesizing Modify the parameter of `ckpt_name` in run.sh file to the weight name after training. Add parameter
`providers=['CUDAExecutionProvider'] `to the file `PaddleSpeech/paddlespeech/t2s/frontend/g2pw/onnx_api.py `at line 80.
Like below:

```sh
self.session_g2pW = onnxruntime.InferenceSession(
    os.path.join(uncompress_path, 'g2pW.onnx'),
    sess_options=sess_options, providers=['CUDAExecutionProvider'])
```

```sh
./run.sh --stage 2 --stop-stage 3
```

### Model Inference

```sh
./run.sh --stage 4 --stop-stage 4
```

## Model Results

| GPUS       | avg_ips             | l1 loss | duration loss | pitch loss | energy loss | loss  |
|------------|---------------------|---------|---------------|------------|-------------|-------|
| BI 100 × 8 | 71.19 sequences/sec | 0.603   | 0.037         | 0.327      | 0.151       | 1.118 |

## References

- [FastSpeech2](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/csmsc/tts3)
