# DPN107
## Model description
A Dual Path Network (DPN) is a convolutional neural network which presents a new topology of connection paths internally.The intuition is that ResNets enables feature re-usage while DenseNet enables new feature exploration, and both are important for learning good representations. To enjoy the benefits from both path topologies, Dual Path Networks share common features while maintaining the flexibility to explore new features through dual path architectures.

## Step 1: Installing
```bash
pip3 install -r requirements.txt
```
:beers: Done!

## Step 2: Training
### Multiple GPUs on one machine (AMP)

Set data path by `export DATA_PATH=/path/to/imagenet`. The following command uses all cards to train:

```bash
bash train_dpn107_amp_dist.sh
```
:beers: Done!

## Reference
- [torchvision](https://github.com/pytorch/vision/tree/main/references/classification)
