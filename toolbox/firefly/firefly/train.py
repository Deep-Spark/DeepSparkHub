# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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
from loguru import logger
import os
import json
from os.path import join
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PrefixTuningConfig, PromptEncoderConfig
from transformers import (
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    Trainer,
    AddedToken
)

import bitsandbytes as bnb
from collections import defaultdict

from firefly.component.collator import PretrainCollator, SFTDataCollator
from firefly.component.dataset import (
    UnifiedSFTDataset,
    ChatGLM2SFTDataset,
    ChatGLM3SFTDataset,
    UnifiedDPODataset
)
from firefly.component.template import template_dict
from firefly.component.argument import QLoRAArguments, PtuningArguments, PrefixArguments, CustomizedArguments
from trl import DPOTrainer, get_kbit_device_map
import datasets
from datasets import load_dataset, concatenate_datasets
from itertools import chain
from tqdm import tqdm
# from component.loss import TargetLMLoss


def verify_model_dtype(model):
    """
    查看模型种各种类型的参数的情况
    """
    dtype2param_num = defaultdict(int)  # 每种数据类型的参数量
    dtype2param_name = defaultdict(list)  # 每种数据类型的参数名称
    dtype2trainable_param_num = defaultdict(int)  # 每种数据类型参与训练的参数量
    dtype2trainable_param_name = defaultdict(list)  # 每种数据类型参与训练的参数名称
    for name, p in model.named_parameters():
        dtype = p.dtype
        dtype2param_num[dtype] += p.numel()
        dtype2param_name[dtype].append(name)
        if p.requires_grad:
            dtype2trainable_param_num[dtype] += p.numel()
            dtype2trainable_param_name[dtype].append(name)
    # 统计全部参数中，各种类型参数分布
    total = 0
    print('verify all params of the model')
    for k, v in dtype2param_num.items():
        total += v
    for k, v in dtype2param_num.items():
        print(k, v, v / total)
    for k, v in dtype2trainable_param_name.items():
        print(k, v)

    print()
    # 统计可训练参数中，各种类型参数分布
    print('verify trainable params the model')
    total_trainable = 0
    for k, v in dtype2trainable_param_num.items():
        total_trainable += v
    for k, v in dtype2trainable_param_num.items():
        print(k, v, v / total_trainable)
    for k, v in dtype2trainable_param_num.items():
        print(k, v)


def find_all_linear_names(model, qlora=False):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    if qlora:
        cls = bnb.nn.Linear4bit
    else:
        cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    lora_module_names = list(lora_module_names)
    logger.info(f'LoRA target module names: {lora_module_names}')
    return lora_module_names


def setup_everything():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_file", type=str, default='train_args/baichuan-sft-qlora.json', help="")
    parser.add_argument("--peft_type", type=str, required=True, help="")
    args = parser.parse_args()
    train_args_file = args.train_args_file
    peft_type = args.peft_type
    # 读取训练的参数配置
    if peft_type == "lora" or peft_type == "qlora":
        parser = HfArgumentParser((QLoRAArguments, TrainingArguments))
    elif peft_type == "ptuning_v2" or peft_type == "prefix_tuning":
        parser = HfArgumentParser((PrefixArguments, TrainingArguments))
    elif peft_type == "ptuning":
        parser = HfArgumentParser((PtuningArguments, TrainingArguments))
    elif peft_type == "full":
        parser = HfArgumentParser((CustomizedArguments, TrainingArguments))
    else:
        raise KeyError
    # 解析得到自定义参数，以及自带参数
    args, training_args = parser.parse_json_file(json_file=train_args_file)
    # 创建输出目录
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    # 加载训练配置文件
    with open(train_args_file, "r") as f:
        train_args = json.load(f)
    # 保存训练参数到输出目录
    with open(join(training_args.output_dir, 'train_args.json'), "w") as f:
        json.dump(train_args, f, indent=4)
    # 设置随机种子
    set_seed(training_args.seed)
    return args, training_args


def load_pretrain_dataset(training_args, args, tokenizer):
    """
    多线程预处理预训练数据
    """
    def tokenize_function(examples):
        output = tokenizer(examples["text"])
        output = {'input_ids': output.input_ids}
        return output

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    data_path = args.train_file
    max_seq_length = args.max_seq_length
    # 创建缓存路径
    cache_dir = join(data_path, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    logger.info('Pretraining data path: {}'.format(data_path))

    # 扫描所有jsonl文件
    logger.info('Scanning all the training file...')
    files = []
    for root, dir_names, file_names in os.walk(data_path):
        for file_name in file_names:
            file = join(root, file_name)
            if file_name.endswith('.jsonl'):
                files.append(file)
    logger.info(f'Total num of training file: {len(files)}')

    # 预处理所有文本，将其id化，并且进行packing操作
    with training_args.main_process_first(desc="dataset map tokenization and grouping"):
        pretrain_dataset = []  # 汇总所有dataset
        for idx, file in enumerate(tqdm(files)):
            logger.info(f'Loading file: {file}')
            file_name = os.path.basename(file)
            file_name = file_name.replace('.jsonl', '')
            cache_path = os.path.join(cache_dir, file_name)
            os.makedirs(cache_path, exist_ok=True)

            try:
                processed_dataset = datasets.load_from_disk(cache_path, keep_in_memory=False)
                logger.info(f'Finished loading datasets-{file_name} from cache')
            except Exception:
                tmp_cache_path = join(cache_path, 'tmp')    # 临时缓存目录，会被自动删除
                logger.info(f'There is no cache of file {file_name}, start preprocessing...')
                raw_dataset = load_dataset("json", data_files=file, cache_dir=tmp_cache_path, keep_in_memory=False)
                tokenized_dataset = raw_dataset.map(
                    tokenize_function,
                    batched=True,
                    num_proc=args.tokenize_num_workers,
                    remove_columns="text",
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    cache_file_names={k: os.path.join(tmp_cache_path, 'tokenized.arrow') for k in raw_dataset},
                    desc="Running tokenizer on dataset",
                )
                grouped_datasets = tokenized_dataset.map(
                    group_texts,
                    batched=True,
                    num_proc=args.tokenize_num_workers,
                    load_from_cache_file=True,
                    keep_in_memory=False,
                    cache_file_names={k: os.path.join(tmp_cache_path, 'grouped.arrow') for k in tokenized_dataset},
                    desc=f"Grouping texts in chunks of {max_seq_length}",
                )
                processed_dataset = grouped_datasets
                processed_dataset.save_to_disk(cache_path)
                # 删除临时目录
                # shutil.rmtree(tmp_cache_path)

            logger.info(f"Training number of {file_name}: {len(processed_dataset['train'])}")
            if idx == 0:
                pretrain_dataset = processed_dataset['train']
            else:
                assert pretrain_dataset.features.type == processed_dataset["train"].features.type
                pretrain_dataset = concatenate_datasets([pretrain_dataset, processed_dataset["train"]])
    logger.info(f"Total training number: {len(pretrain_dataset)}")
    return pretrain_dataset


def load_tokenizer(args):
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # 加载tokenzier
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if config.model_type == 'llama' or config.model_type == 'internlm2' else True
    )

    # 部分模型的base与chat版本的tokenizer存在差异
    if 'internlm2' in args.model_name_or_path.lower():
        tokenizer._added_tokens_encoder.update({'<|im_start|>': 92543})
        tokenizer._added_tokens_encoder.update({'<|im_end|>': 92542})
        tokenizer._added_tokens_decoder.update({92543: AddedToken('<|im_start|>')})
        tokenizer._added_tokens_decoder.update({92542: AddedToken('<|im_end|>')})
        tokenizer.add_special_tokens({'additional_special_tokens': ['<|im_start|>', '<|im_end|>']})
    elif 'orion' in args.model_name_or_path.lower():
        tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>'})
    elif 'gemma' in args.model_name_or_path.lower():
        tokenizer.add_special_tokens({'additional_special_tokens': ['<start_of_turn>', '<end_of_turn>']})

    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    assert tokenizer.eos_token_id is not None, "eos_token_id should not be None"
    logger.info(f'vocab_size of tokenizer: {tokenizer.vocab_size}')
    return tokenizer


def load_model(args, training_args):
    """
    加载模型
    """
    assert training_args.bf16 or training_args.fp16, 'bf16 or fp16 should be True'
    logger.info(f'Loading model from base model: {args.model_name_or_path}')
    logger.info(f'Train model with {args.task_type}')

    # init model kwargs
    # todo add flash attention
    # attn_implementation = None
    torch_dtype = torch.float16 if training_args.fp16 else torch.bfloat16
    if args.peft_type == 'qlora':
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16 if training_args.fp16 else torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    else:
        quantization_config = None
    model_kwargs = dict(
        trust_remote_code=True,
        # attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)

    # moe模型，需要考虑负载均衡的loss
    if 'output_router_logits' in model.config.to_dict():
        logger.info('set output_router_logits as True')
        model.config.output_router_logits = True
    # QLoRA: casts all the non int8 modules to full precision (fp32) for stability
    if args.peft_type == 'qlora' and args.task_type in ['pretrain', 'sft']:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    # LoRA: Enables the gradients for the input embeddings
    elif args.task_type in ['pretrain', 'sft']:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # init peft_config
    if args.peft_type == 'full':
        peft_config = None
    elif args.peft_type == "lora" or args.peft_type == "qlora":
        # 找到所有需要插入adapter的全连接层
        target_modules = find_all_linear_names(model, args.peft_type=="qlora")
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif args.peft_type == "ptuning_v2" or args.peft_type == "prefix_tuning":
        peft_config = PrefixTuningConfig(
            peft_type="PREFIX_TUNING",
            task_type="CAUSAL_LM",
            num_virtual_tokens=args.num_virtual_tokens,
            token_dim=args.token_dim,
            num_attention_heads=args.num_attention_heads,
            encoder_hidden_size=args.encoder_hidden_size,
            prefix_projection=args.prefix_projection,
        )
    elif args.peft_type == "ptuning":
        peft_config = PromptEncoderConfig(
            peft_type="P_TUNING",
            task_type="CAUSAL_LM",
            num_virtual_tokens=args.num_virtual_tokens,
            token_dim=args.token_dim,
            num_attention_heads=args.num_attention_heads,
            num_layers=args.num_layers,
            encoder_reparameterization_type="MLP",
            encoder_hidden_size=args.encoder_hidden_size,
        )
    else:
        raise KeyError("{} is unknow, please check your config".format(args.peft_type))

    # init peft model
    if args.peft_type in ['lora', 'qlora', "ptuning_v2", "ptuning"] and args.task_type in ['pretrain', 'sft']:
        model = get_peft_model(model, peft_config)
        logger.info(f'memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB')
        model.print_trainable_parameters()

    # init ref_model
    if args.task_type == 'dpo':
        ref_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs) if args.peft_type == 'full' else None
    # pretrain和sft，不需要ref_model
    else:
        ref_model = None

    # 计算模型参数量
    total = sum(p.numel() for p in model.parameters())
    logger.info("Total model params: %.2fM" % (total / 1e6))

    return {
        'model': model,
        'ref_model': ref_model,
        'peft_config': peft_config
    }


def load_sft_dataset(args, tokenizer):
    if args.template_name not in template_dict.keys():
        raise Exception(f"template_name doesn't exist, all template_name: {template_dict.keys()}")
    template = template_dict[args.template_name]
    if 'chatglm2' in args.model_name_or_path.lower():
        logger.info('Loading data with ChatGLM2SFTDataset')
        train_dataset = ChatGLM2SFTDataset(args.train_file, tokenizer, args.max_seq_length, template)
    elif 'chatglm3' in args.model_name_or_path.lower():
        logger.info('Loading data with ChatGLM3SFTDataset')
        train_dataset = ChatGLM3SFTDataset(args.train_file, tokenizer, args.max_seq_length, template)
    else:
        logger.info('Loading data with UnifiedSFTDataset')
        train_dataset = UnifiedSFTDataset(args.train_file, tokenizer, args.max_seq_length, template)
    return train_dataset


def load_dpo_dataset(args, tokenizer):
    if args.template_name not in template_dict.keys():
        raise Exception(f"template_name doesn't exist, all template_name: {template_dict.keys()}")
    template = template_dict[args.template_name]
    train_dataset = UnifiedDPODataset(args.train_file, tokenizer, args.max_seq_length, args.max_prompt_length, template)
    return train_dataset


def init_components(args, training_args):
    """
    初始化各个组件
    """
    logger.info('Initializing components...')
    training_args.ddp_find_unused_parameters = False

    # 加载tokenizer
    tokenizer = load_tokenizer(args)
    
    # 加载model
    components = load_model(args, training_args)
    model = components['model']
    ref_model = components['ref_model']
    peft_config = components['peft_config']
    
    # 查看模型种各种类型的参数的情况
    # verify_model_dtype(model)

    # 初始化dataset和collator
    if args.task_type == 'pretrain':
        logger.info('Train model with pretrain task')
        train_dataset = load_pretrain_dataset(training_args, args, tokenizer)
        data_collator = PretrainCollator(tokenizer, args.max_seq_length)
    elif args.task_type == 'sft':
        logger.info('Train model with sft task')
        train_dataset = load_sft_dataset(args, tokenizer)
        data_collator = SFTDataCollator(tokenizer, args.max_seq_length)
    else:
        logger.info('Train model with dpo task')
        train_dataset = load_dpo_dataset(args, tokenizer)
        data_collator = None

    # 初始化Trainer
    if args.task_type == 'dpo':
        trainer = DPOTrainer(
            model,
            ref_model,
            args=training_args,
            beta=args.beta,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            peft_config=peft_config
        )
    # pretrain or sft
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=train_dataset, # fix Trainer: evaluation requires an eval_dataset.
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    return trainer


def main():
    # 进行一些配置和检查
    args, training_args = setup_everything()
    # 加载各种组件
    trainer = init_components(args, training_args)
    # 开始训练
    logger.info("*** starting training ***")
    # todo resume from checkpoint
    # https://github.com/huggingface/transformers/issues/24252
    train_result = trainer.train()
    # 保存最后的checkpoint
    trainer.save_model(training_args.output_dir)  # Saves the tokenizer too
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()


