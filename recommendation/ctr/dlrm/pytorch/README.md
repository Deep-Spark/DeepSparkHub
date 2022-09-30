# DLRM

## Model description

With the advent of deep learning, neural network-based recommendation models have emerged as an important tool for tackling personalization and recommendation tasks. These networks differ significantly from other deep learning networks due to their need to handle categorical features and are not well studied or understood. In this paper, we develop a state-of-the-art deep learning recommendation model (DLRM) and provide its implementation in both PyTorch and Caffe2 frameworks. In addition, we design a specialized parallelization scheme utilizing model parallelism on the embedding tables to mitigate memory constraints while exploiting data parallelism to scale-out compute from the fully-connected layers. We compare DLRM against existing recommendation models and characterize its performance on the Big Basin AI platform, demonstrating its usefulness as a benchmark for future algorithmic experimentation and system co-design.

## Step 1: Installing packages

```shell
$ cd ../../
$ pip3 install -r requirements.txt && python3 ./setup.py install
```


## Step 2: Preparing datasets

Criteo_Terabyte consists of 23 days data, as it is very large, here only take 3 days data for an example.

```shell
$ cd modelzoo/recommendation/ctr/dlrm/pytorch/dlrm/data
$ bash download_and_preprocess.sh
```

After above steps, can get files: terabyte_processed_test.bin, terabyte_processed_train.bin, terabyte_processed_val.bin .



## Step 3: Training

### On single GPU

```shell
$ python3 -u  scripts/train.py --model_config dlrm/config/official_config.json --dataset /home/datasets/recommendation/Criteo_Terabyte  --lr 0.1 --warmup_steps 2750 --decay_end_lr 0 --decay_steps 27772 --decay_start_step 49315 --batch_size 2048 --epochs 5 |& tee 1card.txt
```

### Multiple GPUs on one machine

```shell
$ python3 -u -m torch.distributed.launch --nproc_per_node=8 --use_env scripts/dist_train.py --model_config dlrm/config/official_config.json --dataset /home/datasets/recommendation/Criteo_Terabyte  --lr 0.1 --warmup_steps 2750 --decay_end_lr 0 --decay_steps 27772 --decay_start_step 49315 --batch_size 2048 --epochs 5 |& tee 8cards.txt
```

## Results on BI-V100

| GPUs |  FPS   | AUC  |
| ---- | ------ | ---- |
| 1x1  | 196958 | N/A  |
| 1x8  | 346555 | 0.75 |

| Convergence criteria | Configuration (x denotes number of GPUs) | Performance | Accuracy | Power（W） | Scalability | Memory utilization（G） | Stability |
| -------------------- | ---------------------------------------- | ----------- | -------- | ---------- | ----------- | ----------------------- | --------- |
| AUC:0.75             | SDK V2.2,bs:2048,8x,AMP                  | 793486      | 0.75     | 60\*8      | 0.97        | 3.7\*8                  | 1         |


## Reference
https://github.com/mlcommons/training_results_v0.7/tree/master/NVIDIA/benchmarks/dlrm/implementations/pytorch
