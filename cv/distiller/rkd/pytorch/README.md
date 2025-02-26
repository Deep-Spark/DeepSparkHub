# RKD (Relational Knowledge Distillation)

## Model description

Official implementation of [Relational Knowledge Distillation](https://arxiv.org/abs/1904.05068), CVPR 2019\
This repository contains the source code of experiments for metric learning.

## Step 1: Installation

```bash
# If 'ZLIB_1.2.9' is not found, you need to install it as below.
wget http://www.zlib.net/fossils/zlib-1.2.9.tar.gz
tar xvf zlib-1.2.9.tar.gz
cd zlib-1.2.9/
./configure && make install
cd ..
rm -rf zlib-1.2.9.tar.gz zlib-1.2.9/
```

## Step 2: Distillation

```bash
# Train a teacher embedding network of resnet50 (d=512) sing triplet loss (margin=0.2) with distance-weighted sampling.
python3 run.py --mode train \ 
               --dataset cub200 \
               --base resnet50 \
               --sample distance \ 
               --margin 0.2 \ 
               --embedding_size 512 \
               --save_dir teacher

# Evaluate the teacher embedding network
python3 run.py --mode eval \ 
               --dataset cub200 \
               --base resnet50 \
               --embedding_size 512 \
               --load teacher/best.pth 

# Distill the teacher-to-student embedding network
python3 run_distill.py --dataset cub200 \
                      --base resnet18 \
                      --embedding_size 64 \
                      --l2normalize false \
                      --teacher_base resnet50 \
                      --teacher_embedding_size 512 \
                      --teacher_load teacher/best.pth \
                      --dist_ratio 1  \
                      --angle_ratio 2 \
                      --save_dir student
                      
# Distill the trained model to the student network
python3 run.py --mode eval \ 
               --dataset cub200 \
               --base resnet18 \
               --l2normalize false \
               --embedding_size 64 \
               --load student/best.pth 
```

## Results
| model   | acc |
|:----------:|:--------:|
| RKD|  Best Train Recall: 0.7940, Best Eval Recall: 0.5763 |

## Reference
- [paper](https://arxiv.org/abs/2302.05637)



