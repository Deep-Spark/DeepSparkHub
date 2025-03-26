# CRNN

## Model Description

CRNN (Convolutional Recurrent Neural Network) is an end-to-end trainable model for image-based sequence recognition,
particularly effective for scene text recognition. It combines convolutional layers for feature extraction with
recurrent layers for sequence modeling, followed by a transcription layer. CRNN handles sequences of arbitrary lengths
without character segmentation or horizontal scaling, making it versatile for both lexicon-free and lexicon-based text
recognition tasks. Its compact architecture and unified framework make it practical for real-world applications like
document analysis and OCR.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.0     |  23.09  |

## Model Preparation

### Prepare Resources

- Go to visit [Syn90K official website](https://www.robots.ox.ac.uk/~vgg/data/text/), then download the dataset for
  training. The dataset path structure sholud look like:

```bash
  ├── Syn90k
  │   ├── shuffle_labels.txt
  │   ├── label.txt
  │   ├── label.lmdb
  │   ├── mnt
```

- Go to visit [IIIT5K official
  website](https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset), then download the dataset
  for test. The dataset path structure sholud look like:

```bash
  ├── IIIT5K
  │   ├── traindata.mat
  │   ├── testdata.mat
  │   ├── trainCharBound.mat
  │   ├── testCharBound.mat
  │   ├── lexicon.txt
  │   ├── train
  │   ├── test
```

- The annotation need to be extracted from the matlib data file.

```bash
python3 convert_iiit5k.py -m ./IIIT5K/testdata.mat -o ./IIIT5K -a ./IIIT5K/annotation.txt
```

### Install Dependencies

```shell
# Install requirements
pip3 install -r requirements.txt

# Install openmpi
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.7.tar.gz
tar xf openmpi-4.0.7.tar.gz
cd openmpi-4.0.7/
./configure --prefix=/usr/local/bin --with-orte
make -j4 && make install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
```

## Model Training

```bash
# Run on 1 GPU
python3 train.py --train_dataset=synth --train_dataset_path=./Syn90k/mnt/ramdisk/max/90kDICT32px --device_target=GPU 2>&1 | tee log.txt

# Run on 8 GPU 
mpirun --allow-run-as-root -n 8 --output-filename log_output --merge-stderr-to-stdout \
python3 train.py --train_dataset=synth --train_dataset_path=./Syn90k/mnt/ramdisk/max/90kDICT32px --device_target=GPU --run_distribute=True 2>&1 | tee log.txt

# Run eval
python3 eval.py --eval_dataset=iiit5k \
--eval_dataset_path=./IIIT5K/ \ 
--checkpoint_path=./ckpt_0/crnn-10_14110.ckpt \
--device_target=GPU 2>&1 | tee eval.log
```

## Model Results

| GPUS       | DATASETS | ACC   | FPS     |
|------------|----------|-------|---------|
| BI-V100 x8 | IIIT5K   | 0.798 | 7976.44 |

## References

- [CRNN](https://gitee.com/mindspore/models/tree/master/official/cv/CRNN)
