# Transformer

## Model description
The Transformer architecture, introduced in the paper "Attention Is All You Need" by
Vaswani et al. in 2017, revolutionized the field of deep learning. It relies on a
mechanism called self-attention to process input data in parallel (as opposed to
sequentially) and capture complex dependencies in data, regardless of their distance
in the sequence. Transformers have since become the foundation for state-of-the-art
models in various tasks, especially in natural language processing, such as the BERT
and GPT series.

## Step 1: Installation

```bash
cd ../../../../toolbox/WeNet/
bash install_toolbox_wenet.sh
```

## Step 2: Training

Dataset is data_aishell.tgz and resource_aishell.tgz from wenet.
You could just run the whole script, which will download the dataset automatically.

**You need to modify the path of the dataset in run.sh.**

```bash
# Change to the scripts path
cd wenet/examples/aishell/s0/

# Configure data path and model name
export data_path="/path/to/aishell"
export model_name="transformer"

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

## Results

| GPUs | FP16  | QPS | WER (ctc_greedy_search)| WER (ctc_prefix_beam_search) | WER (attention)| WER (attention_rescoring)|
|---   |---   |---  |---                       |---                            |---              |---                    |
| BI-V100 x8  | False | 394| 5.78%                    | 5.78%                         | 5.59%           | 5.17%                 |

## Reference
- [WeNet](https://github.com/wenet-e2e/wenet)
