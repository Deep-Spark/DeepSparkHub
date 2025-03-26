# DLRM

## Model Description

DLRM (Deep Learning Recommendation Model) is a state-of-the-art neural network architecture designed specifically for
recommendation systems. It effectively handles both categorical and numerical features, making it ideal for
personalization tasks. DLRM employs a unique architecture that combines embedding tables for categorical data with
fully-connected layers for numerical features. Its specialized parallelization scheme uses model parallelism for
embedding tables and data parallelism for dense layers, optimizing memory usage and computational efficiency. DLRM
serves as a benchmark for recommendation system development and performance evaluation.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 2.2.0     |  22.09  |

## Model Preparation

### Prepare Resources

Criteo_Terabyte consists of 23 days data, as it is very large, here only take 3 days data for an example.

```sh
# Check gzip version
gzip -V

# If gzip version is not 1.6+, you need to install gzip 1.6
wget https://ftp.gnu.org/gnu/gzip/gzip-1.6.tar.gz
tar -xzf gzip-1.6.tar.gz
cd gzip-1.6
./configure && make install
cd ../
rm -rf gzip-1.6.tar.gz gzip-1.6/

# Download data
pushd dlrm/data/
bash download_and_preprocess.sh
popd
```

After above steps, you can get files: terabyte_processed_test.bin, terabyte_processed_train.bin,
terabyte_processed_val.bin in "/home/datasets/recommendation/Criteo_Terabyte/".

### Install Dependencies

```sh
pip3 install -r requirements.txt && python3 ./setup.py install
```

## Model Training

```sh
# On single GPU
python3 -u  scripts/train.py --model_config dlrm/config/official_config.json --dataset /home/datasets/recommendation/Criteo_Terabyte  --lr 0.1 --warmup_steps 2750 --decay_end_lr 0 --decay_steps 27772 --decay_start_step 49315 --batch_size 2048 --epochs 5 |& tee 1card.txt

# Multiple GPUs on one machine
python3 -u -m torch.distributed.launch --nproc_per_node=8 --use_env scripts/dist_train.py --model_config dlrm/config/official_config.json --dataset /home/datasets/recommendation/Criteo_Terabyte  --lr 0.1 --warmup_steps 2750 --decay_end_lr 0 --decay_steps 27772 --decay_start_step 49315 --batch_size 2048 --epochs 5 |& tee 8cards.txt
```

## Model Results

| Model | GPUs       | FPS    | AUC  |
|-------|------------|--------|------|
| DLRM  | BI-V100 x1 | 196958 | N/A  |
| DLRM  | BI-V100 x8 | 346555 | 0.75 |

| Convergence criteria | Configuration (x denotes number of GPUs) | Performance | Accuracy | Power（W） | Scalability | Memory utilization（G） | Stability |
|----------------------|------------------------------------------|-------------|----------|------------|-------------|-------------------------|-----------|
| AUC:0.75             | SDK V2.2,bs:2048,8x,AMP                  | 793486      | 0.75     | 60\*8      | 0.97        | 3.7\*8                  | 1         |

## References

- [mlcommons](https://github.com/mlcommons/training_results_v0.7/tree/master/NVIDIA/benchmarks/dlrm/implementations/pytorch)
