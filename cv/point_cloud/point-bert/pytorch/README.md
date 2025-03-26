# Point-BERT

## Model Description

Point-BERT is a new paradigm for learning Transformers to generalize the concept of BERT onto 3D point cloud. Inspired
by BERT, we devise a Masked Point Modeling (MPM) task to pre-train point cloud Transformers. Specifically, we first
divide a point cloud into several local patches, and a point cloud Tokenizer is devised via a discrete Variational
AutoEncoder (dVAE) to generate discrete point tokens containing meaningful local information. Then, we randomly mask
some patches of input point clouds and feed them into the backbone Transformer. The pre-training objective is to recover
the original point tokens at the masked locations under the supervision of point tokens obtained by the Tokenizer.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 2.2.0     |  22.09  |

## Model Preparation

### Prepare Resources

Please refer to [DATASET.md](./DATASET.md) for preparing `ShapeNet55` and `processed ModelNet`.
The dataset dircectory tree would be like:

```bash
data/
├── ModelNet
│   └── modelnet40_normal_resampled
│       ├── modelnet40_test_8192pts_fps.dat
│       └── modelnet40_train_8192pts_fps.dat
├── ScanObjectNN_shape_names.txt
├── ShapeNet55-34
│   ├── ShapeNet-55
│   │   ├── test.txt
│   │   └── train.txt
│   └── shapenet_pc
└── shapenet_synset_dict.json
```

### Install Dependencies

> Warning: Now only support Ubuntu OS. If your OS is centOS, you may need to compile open3d from source.

* system

```bash
apt update
apt install libgl1-mesa-glx
```

* python

```bash
pip3 install argparse easydict h5py matplotlib numpy open3d==0.10 opencv-python pyyaml scipy tensorboardX timm==0.4.5 \
             tqdm transforms3d termcolor scikit-learn==0.24.1 Ninja --default-timeout=1000
```

* Chamfer Distance

```bash
bash install.sh
```

* PointNet++

```bash
cd ./Pointnet2_PyTorch
pip3 install pointnet2_ops_lib/.
cd -
```

* GPU kNN

```bash
pip3 install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

## Model Training

* dVAE train

```bash
bash scripts/train.sh 0 --config cfgs/ShapeNet55_models/dvae.yaml --exp_name dVAE
```

* Point-BERT pre-training

When dVAE has finished training, you should be edit `cfgs/Mixup_models/Point-BERT.yaml`, and add the path of dvae_config-ckpt.

```bash
bash ./scripts/dist_train_BERT.sh <NumGPUs> 12345 --config cfgs/Mixup_models/Point-BERT.yaml --exp_name pointBERT_pretrain --val_freq 2 
```

## References

* [Point-BERT](https://github.com/lulutang0608/Point-BERT)
