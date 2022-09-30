# NCF

## Model description

By replacing the inner product with a neural architecture that can learn an arbitrary function from data, we present a general framework named NCF, short for Neural network-based Collaborative Filtering. NCF is generic and can express and generalize matrix factorization under its framework. To supercharge NCF modelling with non-linearities, we propose to leverage a multi-layer perceptron to learn the user-item interaction function. Extensive experiments on two real-world datasets show significant improvements of our proposed NCF framework over the state-of-the-art methods. Empirical evidence shows that using deeper layers of neural networks offers better recommendation performance.


## Step 1: Installing packages

```
pip3 install -r requirements.txt
```


## Step 2: Preparing datasets

dataset is movielens  

cd /modelzoo/recommendation/collaborative_filtering/ncf/pytorch/  
```
bash download_dataset.sh  

# convert
python3 convert.py --path ./data/ml-20m/ratings.csv --output ./data/ml-20m
```


## Step 3: Training

### Multiple GPUs on one machine

```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
bash run_train_fp32.sh
```

### Multiple GPUs on one machine (AMP)
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
# fp16 train
bash run_train_fp16.sh
```

## Reference
https://github.com/mlcommons/training_results_v0.5/tree/master/v0.5.0/nvidia/submission/code/recommendation/pytorch
