# Efficient Conformer V2 (WeNet)

## Model Description

EfficientFormerV2 mimics MobileNet with its convolutional structure,
offering transformers a series of designs and optimizations for mobile acceleration.
The number of parameters and latency of the model are critical for resource-constrained hardware,
so EfficientFormerV2 combines a fine-grained joint search strategy to propose an efficient network with low latency and size.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.0     |  23.12  |

## Model Preparation

### Install Dependencies

```sh
cd ../../../../toolbox/WeNet/
git clone https://github.com/wenet-e2e/wenet.git
cd wenet/
sed -i 's/^torch/# torch/g' requirements.txt
pip3 install -r requirements.txt

```

## Model Training

Dataset is data_aishell.tgz and resource_aishell.tgz from wenet.
You could just run the whole script, which will download the dataset automatically.

**You need to modify the path of the dataset in run.sh.**

```sh
# Change to the scripts path
cd wenet/examples/aishell/s0/

# Add $data_path and $model_name to run.sh
sed -i s/^data=.*/data=\${data_path}/g run.sh
sed -i s#^train_config=.*#train_config=conf/train_\${model_name}#g run.sh
sed -i s#^dir=.*#dir=exp/\${model_name}#g run.sh

# Configure data path and model name
export data_path="/path/to/aishell"
export model_name="u2++_efficonformer_v2"

# Add torchrun command to PATH
ln -s /usr/local/corex-3.1.0/lib64/python3/dist-packages/bin/torchrun /usr/local/bin/

# Run all stages
bash run.sh --stage -1 --stop-stage 6
```

Or you also run each stage one by one manually and check the result to understand the whole process.  

```sh
# Download data
bash run.sh --stage -1 --stop-stage -1
# Prepare Training data
bash run.sh --stage 0 --stop-stage 0
# Extract optinal cmvn features
bash run.sh --stage 1 --stop-stage 1
# Generate label token dictionary
bash run.sh --stage 2 --stop-stage 2
# Prepare WeNet data format
bash run.sh --stage 3 --stop-stage 3
# Neural Network training
bash run.sh --stage 4 --stop-stage 4
# Recognize wav using the trained model
bash run.sh --stage 5 --stop-stage 5
# Export the trained model
bash run.sh --stage 6 --stop-stage 6
```

## Model Results

| GPUs       | QPS | WER(ctc_greedy_search) | WER(ctc_prefix_beam_search) | WER(attention) | WER(attention_rescoring) |
|------------|-----|------------------------|-----------------------------|----------------|--------------------------|
| BI-V100 x8 | 234 | 5.00%                  | 4.99%                       | 4.89%          | 4.58%                    |

## References

- [WeNet](https://github.com/wenet-e2e/wenet)
