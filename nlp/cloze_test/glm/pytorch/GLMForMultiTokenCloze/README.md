# GLM

## Model description

There have been various types of pretraining architectures including autoencoding models (e.g., BERT), autoregressive models (e.g., GPT), and encoder-decoder models (e.g., T5). However, none of the pretraining frameworks performs the best for all tasks of three main categories including natural language understanding (NLU), unconditional generation, and conditional generation. We propose a General Language Model (GLM) based on autoregressive blank infilling to address this challenge. GLM improves blank filling pretraining by adding 2D positional encodings and allowing an arbitrary order to predict spans, which results in performance gains over BERT and T5 on NLU tasks. Meanwhile, GLM can be pretrained for different types of tasks by varying the number and lengths of blanks. On a wide range of tasks across NLU, conditional and unconditional generation, GLM outperforms BERT, T5, and GPT given the same model sizes and data, and achieves the best performance from a single pretrained model with 1.25x parameters of BERT Large , demonstrating its generalizability to different downstream tasks.

## Step 1: Installing packages

```
$ bash prepare_environment.sh
```

## Step 2: Preparing data

```
$ cd nlp/cloze_test/glm/pytorch/GLMForMultiTokenCloze/base
$ bash preparedata.sh /home/data/perf/glm
```

## Step 3: Preparing data prepare pretrained weights

download from [glm github](https://github.com/THUDM/GLM)
[model release page](https://mailstsinghuaeducn-my.sharepoint.com/personal/duzx16_mails_tsinghua_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fduzx16%5Fmails%5Ftsinghua%5Fedu%5Fcn%2FDocuments%2Fmodels&ga=1) download `glm-large-blank.tar.bz2`

```
$ cd /home/data/perf/glm
$ tar -jxvf glm-large-blank.tar.bz2
```


## Step 4: Training
   
### Multiple GPUs on one machine

 ```bash
 $ cd nlp/cloze_test/glm/pytorch/GLMForMultiTokenCloze/base
 $ bash run.sh
 ```

## Results on BI-V100

| GPUs | Batch Size | FPS | Accuracy |
|------|------------| --- | ------------ |
| 1x8  | 8          | 9.43 | 0.81         |


## Reference
https://github.com/THUDM/GLM