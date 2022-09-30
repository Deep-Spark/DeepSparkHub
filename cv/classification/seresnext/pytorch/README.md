# SEResNeXt
## Model description
SE ResNeXt is a variant of a ResNext that employs squeeze-and-excitation blocks to enable the network to perform dynamic channel-wise feature recalibration.
## Step 1: Installing
```bash
pip3 install -r requirements.txt
```
:beers: Done!

## Step 2: Training
### Multiple GPUs on one machine
Set data path by `export DATA_PATH=/path/to/imagenet`. The following command uses all cards to train:

```bash
bash train_seresnext101_32x4d_amp_dist.sh
```

:beers: Done!


## Reference
https://github.com/osmr/imgclsmob/blob/f2993d3ce73a2f7ddba05da3891defb08547d504/pytorch/pytorchcv/models/seresnext.py#L214
