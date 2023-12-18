# BERT Question Answering

## Model description

BERT-base SQuAD task Fine-tuning

## Step 1: Installation

```bash
cd  <your_project_path>/nlp/querstion_answering/bert/pytorch
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

```bash
## config "dataset_name" to be local path of squad if you want to run off-line
## e.g. --dataset_name /path/to/squad
bash run.sh
```

### Multiple GPUs on one machine

```bash
## config "dataset_name" to be local path of squad if you want to run off-line
## e.g. --dataset_name /path/to/squad
bash run_dist.sh
```
## Results on BI-V100

| GPUs | Samples/s | F1     |
|------|-----------|--------|
| 1x1  | 128.86    | 87     |
| 1x8  | 208.6     | 78.69  |

## Reference
https://github.com/huggingface/
