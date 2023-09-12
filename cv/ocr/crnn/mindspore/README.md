# CRNN

## Model description

CRNN was a neural network for image based sequence recognition and its Application to scene text recognition.In this paper, we investigate the problem of scene text recognition, which is among the most important and challenging tasks in image-based sequence recognition. A novel neural network architecture, which integrates feature extraction, sequence modeling and transcription into a unified framework, is proposed. Compared with previous systems for scene text recognition, the proposed architecture possesses four distinctive properties: (1) It is end-to-end trainable, in contrast to most of the existing algorithms whose components are separately trained and tuned. (2) It naturally handles sequences in arbitrary lengths, involving no character segmentation or horizontal scale normalization. (3) It is not confined to any predefined lexicon and achieves remarkable performances in both lexicon-free and lexicon-based scene text recognition tasks. (4) It generates an effective yet much smaller model, which is more practical for real-world application scenarios.

[Paper](https://arxiv.org/abs/1507.05717): Baoguang Shi, Xiang Bai, Cong Yao, "An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition", ArXiv, vol. abs/1507.05717, 2015.

## Step 1:Installation

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

## Step 2:Preparing datasets

* Go to visit [Syn90K official website](https://www.robots.ox.ac.uk/~vgg/data/text/), then download the dataset for training. The dataset path structure sholud look like:

  ```
  ├── Syn90k
  │   ├── shuffle_labels.txt
  │   ├── label.txt
  │   ├── label.lmdb
  │   ├── mnt
  ```
* Go to visit [IIIT5K official website](https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset), then download the dataset for test. The dataset path structure sholud look like:

```
├── IIIT5K
│   ├── traindata.mat
│   ├── testdata.mat
│   ├── trainCharBound.mat
│   ├── testCharBound.mat
│   ├── lexicon.txt
│   ├── train
│   ├── test
```
* The annotation need to be extracted from the matlib data file.
```
python3 convert_iiit5k.py -m ./IIIT5K/testdata.mat -o ./IIIT5K -a ./IIIT5K/annotation.txt
```
## Step 3:Training

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

## Results

| GPUS       | DATASETS   | ACC     |  FPS    | 
| ---------- | ---------- | ------  | ------  |
| BI-V100 x8 | IIIT5K     | 0.798   | 7976.44 |

## Reference

[CRNN](https://gitee.com/mindspore/models/tree/master/official/cv/CRNN)
