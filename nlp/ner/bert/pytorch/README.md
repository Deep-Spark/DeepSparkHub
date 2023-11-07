# BERT NER

## Model description

BERT-base NER task Fine-tuning

## Step 1: Installing packages

``` shell
cd  <your_project_path>/nlp/ner/bert/pytorch
pip3 install -r requirements.txt
```

## Step 2: Preparing datasets

```bash
# Get "bert-base-uncased" from [Huggingface](https://huggingface.co/bert-base-uncased)

## Install lfs
wget https://packagecloud.io/github/git-lfs/packages/el/7/git-lfs-2.13.2-1.el7.x86_64.rpm/download -O lfs.rpm
rpm -ivh lfs.rpm

## Clone from Huggingface, it may take long time for large file
git lfs install
git clone https://huggingface.co/bert-base-uncased
```


## Step 3: Training

**Make sure you've got "bert-base-uncased" ready in ./bert-base-uncased**

### On single GPU

``` shell
## config "dataset_name" to be local path of conll2003 if you want to run off-line
bash run.sh
```

### Multiple GPUs on one machine

```shell
## config "dataset_name" to be local path of conll2003 if you want to run off-line
bash run_dist.sh
```
## Results on BI-V100

| GPUs | Samples/s  | Loss |
|------|------|----|
| 1x1  | 100 | 0.0696 |
| 1x8  | 252 | 0.0688 |

## Reference
https://github.com/huggingface/
