# LLaMa2 7B Reward Model Finetuning

## Model description
LLaMA2 is a large language model released by Meta in 2023, with parameters ranging from 7B to 70B. Compared to LLaMA, the training corpus of LLaMA2 is 40% longer, and the context length has been upgraded from 2048 to 4096, allowing for understanding and generating longer texts. 

## Step 1: Prepare

Prepare datasets and pretrained model weight

```
 $ mkdir -p /home/datasets/nlp/Dahoas && cd /home/datasets/nlp/Dahoas
 $ git clone https://huggingface.co/datasets/Dahoas/rm-static

  get Llama-2-7b-hf from huggingface models or aifasthub ( https://aifasthub.com/models/NousResearch/Llama-2-7b-hf ).
 
 $ mkdir -p /home/model_zoo/nlp/ && mv Llama-2-7b-hf /home/model_zoo/nlp/
```

Prepare training environment
```
$ cd /path/deepsparkhub/nlp/llm/llama2-7b_reward_sft/DeepSpeedExamples/applications/DeepSpeed-Chat/
$ pip install -r requirements.txt
$ pip install -e .
```

## Step 2: Run

Fine-tune

```
$ cd training/step2_reward_model_finetuning/
$ bash ./run_llama2_7b.sh
```

## Results
| GPUs        | SamplesPerSec   | ACC          |
|-------------|-----------------|--------------|
| BI-V100 x 8 | 2.726           | 0.68         |


## Reference
- [Reference_link] (https://github.com/microsoft/DeepSpeedExamples/)
