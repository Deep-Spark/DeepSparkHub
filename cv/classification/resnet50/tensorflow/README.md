
## Prepare

### Install packages

```shell
pip3 install absl-py git+https://github.com/NVIDIA/dllogger#egg=dllogger
```

### Download datasets

```shell

wget http://10.150.9.95/swapp/datasets/cv/classification/imagenette_tfrecord.tgz

tar -xzvf imagenette_tfrecord.tgz
rm -rf imagenette_tfrecord.tgz
```

## Training

### Training on single card

```shell
bash run_train_resnet50_imagenette.sh
```

### Training on mutil-cards
```shell
bash run_train_resnet50_multigpu_imagenette.sh
```
