# GLM

## Model Description

GLM (General Language Model) is a versatile pretraining architecture that unifies natural language understanding and
generation tasks. Unlike traditional models limited to specific paradigms, GLM employs autoregressive blank infilling
with 2D positional encodings, enabling flexible span prediction. This approach allows GLM to excel across diverse tasks
including NLU, conditional generation, and unconditional generation. With its ability to adapt to various downstream
tasks through adjustable blank configurations, GLM outperforms specialized models like BERT, T5, and GPT while
maintaining efficiency.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 2.2.0     |  22.09  |

## Model Preparation

### Prepare Resources

```bash
bash preparedata.sh /home/data/perf/glm
```

Download `glm-large-blank.tar.bz2` from [Tsinghua-Cloud](https://cloud.tsinghua.edu.cn/d/13f5b03da9594e5490c4/).

```bash
mkdir -p /home/data/perf/glm
pushd /home/data/perf/glm
tar -jxvf glm-large-blank.tar.bz2
popd
```

### Install Dependencies

```bash
cd nlp/cloze_test/glm/pytorch/GLMForMultiTokenCloze/base
bash prepare_environment.sh
```

## Model Training

```bash
# Multiple GPUs on one machine
bash run.sh
```

## Model Results

| GPUs       | Batch Size | FPS  | Accuracy |
|------------|------------|------|----------|
| BI-V100 x8 | 8          | 9.43 | 0.81     |

## References

- [GLM](https://github.com/THUDM/GLM)
