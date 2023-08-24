# Conformer

## Model description
The Conformer neural network is a hybrid model designed for tasks like speech recognition,
merging the strengths of convolutional neural networks (CNNs) and transformers. It employs
CNNs for local feature extraction and transformers to capture long-range dependencies in
data. This combination allows the Conformer to efficiently handle both local patterns and
global relationships, making it particularly effective for audio and speech tasks.

## Step 1: Installation

```bash
cd ../../../../toolbox/WeNet/
bash install_toolbox_wenet.sh
```

## Step 2: Training

Dataset is data_aishell.tgz and resource_aishell.tgz from wenet.
You could just run the whole script, which will download the dataset automatically.

**You need to modify the path of dataset in run.sh.**

```bash
# Change to the scripts path
cd wenet/examples/aishell/s0/

# Configure data path and model name
export data_path="/path/to/aishell"
export model_name="conformer"

# Run all stages
bash run.sh --stage -1 --stop-stage 6
```

Or you also run each stage one by one manually and check the result to understand the whole process.  

```bash
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

## Results on BI-V100

| GPUs | FP16  | QPS |
|------|-------|-----|
| 1x8  | False | 67  |
| 1x8  | True  | 288 |

| Convergence criteria | Configuration (x denotes number of GPUs) | Performance | Accuracy | Power（W） | Scalability | Memory utilization（G） | Stability |
|----------------------|------------------------------------------|-------------|----------|------------|-------------|-------------------------|-----------|
| 3.72                 | SDK V2.2,bs:32,8x,fp32                   | 380         | 4.79@cer | 113\*8     | 0.82        | 21.5\*8                 | 1         |

## Reference
https://github.com/wenet-e2e/wenet
