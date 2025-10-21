# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.


import argparse

from random import randrange
import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, TaskType, PeftModel, AutoPeftModelForCausalLM

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
    pipeline
)
from trl import SFTTrainer
from models.modeling_phi3 import Phi3ForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", help="model_id in HF")
    parser.add_argument("--dataset_name", type=str, default="", help="Path to the training data")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split")
    parser.add_argument("--lora_r", type=int, default=16, help="dimension of the LoRA attention")
    parser.add_argument("--lora_alpha", type=int, default=16, help="alpha of the LoRA attention")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="dropout of the LoRA attention")
    parser.add_argument("--use_4bit", action="store_true", help="use 4bit quantization")
    parser.add_argument("--bnb_4bit_use_double_quant", action="store_true", help="use double quantization")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", help="quantization type of the 4bit quantization")
    parser.add_argument("--target_modules", type=str, default=["k_proj,q_proj,v_proj,o_proj"], help="modules to replace with LoRA")
    parser.add_argument("--batch_size", type=int, default=4, help="batch_size")
    parser.add_argument("--output_dir", type=str, default="output", help="output directory")
    return parser.parse_args()

def main():
    set_seed(1234)
    cfg = parse_args()
    # prepare dataset
    dataset = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    print(f"dataset size: {len(dataset)}")

    # tokenizer for dataset processing
    tokenizer_id = cfg.model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.padding_side = 'right'
    
    def create_message_column(row):
        messages = []
        user = {
            "content": f"{row['instruction']}\n Input: {row['input']}",
            "role": "user"
        }
        messages.append(user)
        assistant = {
            "content": f"{row['output']}",
            "role": "assistant"
        }
        messages.append(assistant)
        return {"messages": messages}

    def format_dataset_chatml(row):
        return {"text": tokenizer.apply_chat_template(row["messages"], add_generation_prompt=False, tokenize=False)}
    
    dataset_chatml = dataset.map(create_message_column)
    dataset_chatml = dataset_chatml.map(format_dataset_chatml)
    dataset_chatml = dataset_chatml.train_test_split(test_size=0.05, seed=1234)
    
    # prepare model
    if torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        attn_implementation = 'flash_attention_2'
    else:
        compute_dtype = torch.float16
        attn_implementation = 'sdpa'
    print(compute_dtype)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True, add_eos_token=True, use_fast=True)

    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'left'
    
    if cfg.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=cfg.use_4bit,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant)
    else:
        bnb_config = None

    #model = AutoModelForCausalLM.from_pretrained(
    model = Phi3ForCausalLM.from_pretrained(
            cfg.model_id, torch_dtype=compute_dtype, trust_remote_code=True, quantization_config=bnb_config,
            attn_implementation=attn_implementation)
    
    args = TrainingArguments(
            output_dir=cfg.output_dir,
            evaluation_strategy="steps",
            do_eval=True,
            optim="adamw_torch",
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=8,
            log_level="debug",
            save_strategy="epoch",
            logging_steps=100,
            learning_rate=1e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            eval_steps=100,
            num_train_epochs=1,
            # max_steps=10,
            warmup_ratio=0.1,
            lr_scheduler_type="linear",
            report_to=None,
            seed=42,)

    peft_config = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                task_type=TaskType.CAUSAL_LM,
                # target_modules=cfg.target_modules)
                target_modules=['qkv_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"])
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_chatml['train'],
        eval_dataset=dataset_chatml['test'],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=args)
    
    trainer.train()
    trainer.save_model()
    
    del model
    del trainer

    import gc
    torch.cuda.empty_cache()
    gc.collect()
    
    new_model = AutoPeftModelForCausalLM.from_pretrained(
            args.output_dir,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.bfloat16, #torch.float16,
            trust_remote_code=True,)
    merged_model = new_model.merge_and_unload()
    merged_model.save_pretrained("merged_model", trust_remote_code=True, safe_serialization=True)
    tokenizer.save_pretrained("merged_model")
    

if __name__ == "__main__":
    main()
