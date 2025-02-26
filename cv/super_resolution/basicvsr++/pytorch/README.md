# basicVSR++ (CVPR2022, Oral)

## Model description

A recurrent structure is a popular framework choice for the task of video super-resolution. The state-of-the-art method BasicVSR adopts bidirectional propagation with feature alignment to effectively exploit information from the entire input video. In this study, we redesign BasicVSR by proposing second-order grid propagation and flow-guided deformable alignment. We show that by empowering the recurrent framework with the enhanced propagation and alignment, one can exploit spatiotemporal information across misaligned video frames more effectively. The new components lead to an improved performance under a similar computational constraint. In particular, our model BasicVSR++ surpasses BasicVSR by 0.82 dB in PSNR with similar number of parameters. In addition to video super-resolution, BasicVSR++ generalizes well to other video restoration tasks such as compressed video enhancement. In NTIRE 2021, BasicVSR++ obtains three champions and one runner-up in the Video Super-Resolution and Compressed Video Enhancement Challenges. Codes and models will be released to MMEditing.

## Step 1: Installing packages

```shell
sh build_env.sh
```

## Step 2: Preparing datasets

Download REDS dataset from [homepage](https://seungjunnah.github.io/Datasets/reds.html)
```shell
mkdir -p data/
ln -s ${REDS_DATASET_PATH} data/REDS
```

## Step 3: Training

### One single GPU
```shell
python3 train.py <config> [training args]    # config file can be found in the configs directory
```

### Mutiple GPUs on one machine
```shell
bash dist_train.sh <config file> <num_gpus> [training args]    # config file can be found in the configs directory 
```
### Example

```shell
bash dist_train.sh configs/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4.py 8
```

## Reference
https://github.com/open-mmlab/mmediting
