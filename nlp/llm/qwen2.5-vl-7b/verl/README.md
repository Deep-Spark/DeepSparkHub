# Qwen2.5-VL-7B grpo (verl)

## Model Description

Qwen2.5-VL is not only proficient in recognizing common objects such as flowers, birds, fish, and insects, but it is highly capable of analyzing texts, charts, icons, graphics, and layouts within images.
Directly plays as a visual agent that can reason and dynamically direct tools, which is capable of computer use and phone use. Can comprehend videos of over 1 hour, and this time it has a new ability of cpaturing event by pinpointing the relevant video segments. Can accurately localize objects in an image by generating bounding boxes or points, and it can provide stable JSON outputs for coordinates and attributes. For data like scans of invoices, forms, tables, etc. Qwen2.5-VL supports structured outputs of their contents, benefiting usages in finance, commerce, etc.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| BI-V150 | 4.3.0     |  25.07  |

## Environment Preparation

### Install Dependencies
```bash
cd toolbox/verl/v0.5.0
pip3 install -r requirements.txt
python3 setup.py install
```

### Prepare Resources

```bash
python3 examples/data_preprocess/geo3k.py
mv ~/data/geo3k /home/datasets/verl/geo3k

# Download Qwen2.5-VL-7B-Instruct and put to /home/model_zoos/verl/Qwen2.5-VL-7B-Instruct

```

## Model Training

### train on gsm8k
```bash
cd nlp/llm/qwen2.5-vl-7b/verl
bash run_qwen2_5_vl_7B_geo3k.sh
```

## Model Results

## References

- [verl](https://github.com/volcengine/verl/tree/v0.5.0)
