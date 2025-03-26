# RKD

## Model Description

RKD (Relational Knowledge Distillation) is a knowledge distillation technique that transfers relational information
between data points from a teacher model to a student model. Instead of mimicking individual outputs, RKD focuses on
preserving the relationships (distance and angle) between embeddings. This approach is particularly effective for metric
learning tasks, where maintaining the relative structure of the embedding space is crucial. RKD enhances student model
performance by capturing higher-order relational knowledge from the teacher.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.0.0     |  23.06  |

## Model Preparation

### Install Dependencies

```bash
# If 'ZLIB_1.2.9' is not found, you need to install it as below.
wget http://www.zlib.net/fossils/zlib-1.2.9.tar.gz
tar xvf zlib-1.2.9.tar.gz
cd zlib-1.2.9/
./configure && make install
cd ..
rm -rf zlib-1.2.9.tar.gz zlib-1.2.9/
```

## Model Training

### Model Distillation

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

## Model Results

| Model | ACC                                                 |
|-------|-----------------------------------------------------|
| RKD   | Best Train Recall: 0.7940, Best Eval Recall: 0.5763 |

## References

- [Paper](https://arxiv.org/abs/2302.05637)
