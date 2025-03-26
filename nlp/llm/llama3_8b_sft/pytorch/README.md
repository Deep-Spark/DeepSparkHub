# Llama3-8B SFT (ColossalAI)

## Model Description

Llama3-8B SFT is a fine-tuned version of Meta's Llama3-8B model, optimized using supervised fine-tuning techniques. With
8 billion parameters, it leverages an advanced transformer architecture and Grouped-Query Attention (GQA) for efficient
inference. The SFT process enhances its performance on specific tasks by leveraging labeled datasets, making it
particularly effective for applications requiring precise language understanding and generation. This model combines the
foundational capabilities of Llama3 with task-specific optimizations, offering improved performance while maintaining
computational efficiency.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V150 | 4.1.1     |  24.12  |

## Model Preparation

### Prepare Resources

Get "Meta-Llama-3-8B" models and config file from modelscope or other place, and mv it to "/home/model_zoos/".
One recommended link: "<https://modelscope.cn/models/LLM-Research/Meta-Llama-3-8B>".

```sh
# Prepare ColossalAI
git clone -b v0.4.4 https://github.com/hpcaitech/ColossalAI.git --depth=1
cd ColossalAI/
cp -rf <DeepSparkHub_Root>/toolbox/ColossalAI/v0.4.4/patches/* ./

# Get Meta-Llama-3-8B
mkdir -p /home/model_zoos/
mv <Path>/Meta-Llama-3-8B /home/model_zoos/

wget http://files.deepspark.org.cn:880/deepspark/data/tokenizer/tokenizer.model
cp tokenizer.model /home/model_zoos/Meta-Llama-3-8B

# Get school_math_0.25M.jsonl
wget http://files.deepspark.org.cn:880/deepspark/data/datasets/school_math_0.25M.jsonl
mkdir -p dataset/school_math/convert/
mv school_math_0.25M.jsonl dataset/school_math
bash ./prepare_sft_dataset.sh llama3
```

### Install Dependencies

You should ensure that the corresponding version of ColossalAI has been installed in the iluvatar environment. Then
install applications as follows:

```sh
cd applications/Colossal-LLaMA/
pip3 install -e . 
```

## Model Training

```sh
bash run_llama3_8b_sft_3d.sh
```

## Model Results

| Model     | peft     | num_gpus | train_samples_per_second |
|-----------|----------|----------|--------------------------|
| Llama3-8B | Full sft | 16       | 1.53                     |

## References

- [ColossalAI](https://github.com/hpcaitech/ColossalAI/tree/v0.4.4/applications/Colossal-LLaMA)
