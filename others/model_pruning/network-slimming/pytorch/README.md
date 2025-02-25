# Network Slimming (Pytorch)

## Model description
This repository contains an official PyTorch implementation for the following paper  
[Learning Efficient Convolutional Networks Through Network Slimming](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html) (ICCV 2017).  
Original implementation: [slimming](https://github.com/liuzhuang13/slimming) in Torch.    
The code is based on [pytorch-slimming](https://github.com/foolwood/pytorch-slimming). I add support for ResNet and DenseNet.  

## Train with Sparsity and Using apex mixed precision training

```bash
pip3 install matplotlib
```

```bash
python3 main.py -sr -amp_loss --s 0.0001 --dataset cifar10 --arch vgg --depth 19 --filename vgg --epochs 160

python3 main.py -sr -amp_loss --s 0.00001 --dataset cifar10 --arch resnet --depth 164 --filename resnet --epochs 160

python3 main.py -sr -amp_loss --s 0.00001 --dataset cifar10 --arch densenet --depth 40 --filename densenet --epochs 160
```

## Prune

```bash
python3 vggprune.py --dataset cifar10 --depth 19 --percent 0.7 --model [PATH TO THE MODEL] --filename vgg_prune

python3 resprune.py --dataset cifar10 --depth 164 --percent 0.6 --model [PATH TO THE MODEL] --filename resnet_prune

python3 denseprune.py --dataset cifar10 --depth 40 --percent 0.6 --model [PATH TO THE MODEL] --filename densenet_prune
```

## Fine-tune

```bash
python3 main.py -amp_loss --refine [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch vgg --depth 19 --epochs 160 --filename pruned_vgg

python3 main.py -amp_loss --refine [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch resnet --depth 164 --epochs 160 --filename pruned_resnet

python3 main.py -amp_loss --refine [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch densenet --depth 40 --epochs 160 --filename pruned_densenet
```

## Results

> The results are fairly close to the original paper, whose results are produced by Torch. Note that due to different random seeds, there might be up to ~0.5%/1.5% fluctation on CIFAR-10/100 datasets in different runs, according to our experiences.

### CIFAR10
|  CIFAR10-Vgg  |  Sparsity (1e-4) | Prune (70%) | Fine-tune-160(70%) |
| :---------------: |:--------------------------: | :-----------------: | :-------------------: |
| Top1 Accuracy (%) |            92.0            |        91.9        |         92.4         |
|    Parameters     |            20.04M            |        2.25M        |         2.25M         |

|  CIFAR10-Resnet-164  |    Sparsity (1e-5)  |   Prune(60%)     |  Fine-tune-160(60%)       |
| :---------------: | :-------------------: |  :----------------:| :--------------------:|
| Top1 Accuracy (%) |            93.9             |      13.8       |     94.3     |
|    Parameters     |             1.73M            |      1.12M          |   1.12M           |

|  CIFAR10-Densenet-40  |  Sparsity (1e-5) |       Prune(60%)   | Fine-tune-160(60%) |
| :---------------: | :-------------------: | :--------------------: | :-----------------:|
| Top1 Accuracy (%) |           92.8             |      11.6       |     93.27     |
|    Parameters     |            1.07M            |       0.49M      |    0.49M     |

## Reference
- [network-slimming](https://github.com/Eric-mingjie/network-slimming)
- [paper](https://arxiv.org/abs/1708.06519)
