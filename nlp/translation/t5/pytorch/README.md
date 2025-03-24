# T5

## Model Description

T5 (Text-to-Text Transfer Transformer) is a versatile transformer-based model that approaches all NLP tasks through a
unified text-to-text framework. It treats every task, including translation, summarization, and question answering, as a
text generation problem. This allows T5 to use the same architecture and training procedure across diverse applications.
By converting inputs and outputs into text sequences, T5 demonstrates strong performance across multiple benchmarks
while maintaining a consistent and scalable approach to natural language processing tasks.

## Model Preparation

### Install Dependencies

``` shell
cd  <your_project_path>/t5/pytorch
bash examples_ix/init_torch.sh
```

## Model Training

``` shell
# On single GPU
bash examples_ix/train_t5_small_torch.sh

# On single GPU (AMP)
bash examples_ix/train_t5_small_amp_torch.sh

# Multiple GPUs on one machine
bash examples_ix/train_t5_small_dist_torch.sh

# Multiple GPUs on one machine (AMP)
bash examples_ix/train_t5_small_amp_dist_torch.sh
```

## Model Results

| Model | GUSs       | Samples/s | Loss |
|-------|------------|-----------|------|
| T5    | BI-V100 x1 | 339       | 1.18 |
| T5    | BI-V100 x8 | 2488      | 1.18 |

## References

- [t5-small](https://huggingface.co/google-t5/t5-small)
