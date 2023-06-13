# Transformer

## Step 1: Installing packages

```
pip3 install -r requirements.txt
```

## Step 2: Training

Dataset is data_aishell.tgz and resource_aishell.tgz from wenet.
You could just run the whole script, which will download the dataset automatically.


You need to modify the path of dataset in run.sh.
```
bash run.sh --stage -1 --stop-stage 6
```
Or you also run each stage one by one manually and check the result to understand the whole process.  
```
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

| GPUs | FP16  | QPS | WER (ctc_greedy_search )| WER (ctc_prefix_beam_search  ) | WER (attention )| WER (attention_rescoring )|
|---    |---   |---  |---                       |---                            |---              |---                        |
| 1x8  | False | 394| 5.78%                    | 5.78%                         | 5.59%           | 5.17%                    | 




## Reference
https://github.com/wenet-e2e/wenet
