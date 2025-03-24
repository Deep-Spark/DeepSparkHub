# DeepFM

## Model Description

DeepFM (Deep Factorization Machine) combines Factorization Machines (FM) and Deep Neural Networks (DNN) for
recommendation systems. FM captures low-order feature interactions, while DNN models high-order non-linear interactions.
The model is end-to-end trainable and excels in tasks like click-through rate (CTR) prediction and personalized
recommendations. By integrating both FM and DNN, DeepFM efficiently handles sparse data, offering better performance
compared to traditional methods, especially in large-scale applications such as advertising and product recommendations.

## Model Preparation

### Prepare Resources

```sh
# Prepare PaddleRec
git clone -b release/2.3.0  https://github.com/PaddlePaddle/PaddleRec.git
cd PaddleRec/

# Prepare Criteo dataset
pushd datasets/criteo/
sh run.sh
popd
```

### Install Dependencies

```sh
pip3 install -r requirements.txt
```

## Model Training

```sh
cd models/rank/deepfm
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=3
# train
python3 -u ../../../tools/trainer.py -m config_bigdata.yaml
# Eval
python3 -u ../../../tools/infer.py -m config_bigdata.yaml
```

## References

- [PaddleRec](https://github.com/PaddlePaddle/PaddleRec.git)
