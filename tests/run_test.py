# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import yaml
import subprocess
import json
import re
import time
import logging
import os
import sys
import argparse

def is_debug():
    is_debug_flag = os.environ.get("IS_DEBUG")
    if is_debug_flag and is_debug_flag.lower()=="true":
        return True
    else:
        return False

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model", type=str, help="model name, e.g: alexnet")
    args = parser.parse_args()

    if args.model:
        test_model = args.model
    else:
        test_model = os.environ.get("TEST_CASE")
        test_framework = os.environ.get("FRAMEWORK")
        test_category = os.environ.get("category")
    logging.info(f"Test case to run: {test_model}")
    if not test_model:
        logging.error("test model case is empty")
        sys.exit(-1)
    
    model = get_model_config(test_model, test_framework, test_category)
    if not model:
        logging.error("mode config is empty")
        sys.exit(-1)

    result = {}
    if model["category"] == "cv/classification" and model["framework"] == "pytorch":
        logging.info(f"Start running {model['model_name']} test case:\n{json.dumps(model, indent=4)}")
        result = run_clf_testcase(model)
        check_model_result(result)
        logging.debug(f"The result of {model['model_name']} is\n{json.dumps(result, indent=4)}")
        logging.info(f"End running {model['model_name']} test case.")

    if model["category"] == "cv/detection" and (model["framework"] == "pytorch" or model["framework"] == "paddlepaddle"):
        logging.info(f"Start running {model['model_name']} test case:\n{json.dumps(model, indent=4)}")
        result = run_detec_testcase(model)
        check_model_result(result)
        logging.debug(f"The result of {model['model_name']} is\n{json.dumps(result, indent=4)}")
        logging.info(f"End running {model['model_name']} test case.")

    if model["category"] == "nlp/llm" and model["framework"] == "pytorch":
        logging.info(f"Start running {model['model_name']} test case:\n{json.dumps(model, indent=4)}")
        result = run_llm_testcase(model)
        check_model_result(result)
        logging.debug(f"The result of {model['model_name']} is\n{json.dumps(result, indent=4)}")
        logging.info(f"End running {model['model_name']} test case.")

    logging.info(f"Full text result: {result}")

def get_model_config(model_name, framework, category):
    print(f"model_name: {model_name}, framework: {framework}, category: {category}")
    with open('all_deepsparkhub_model_info.json', mode='r', encoding='utf-8') as config_file:
        config_data = json.load(config_file)

    for model in config_data["models"]:
        # TODO: 模型名称+模型框架+模型类别才能唯一确定一个模型
        if model["model_name"] == model_name.lower() and model["framework"] == framework.lower() and model["category"] == category.lower():
            return model
    return

def check_model_result(result):
    status = "PASS"
    for prec in ["fp16", "int8"]:
        if prec in result["result"]:
            if result["result"][prec]["status"] == "FAIL":
                status = "FAIL"
                break
    result["status"] = status

def run_detec_testcase(model):
    model_name = model["model_name"]
    result = {
        "name": model_name,
        "framework": model["framework"],
        "toolbox": model["toolbox"],
        "category": "cv/detection",
        "result": {},
    }
    is_mmdetection = True if model["toolbox"] == "mmdetection" else False
    is_yolov = True if model["toolbox"] == "yolov" else False
    is_paddledetection = True if model["toolbox"] == "PaddleDetection" else False
    deepsparkhub_path = model["deepsparkhub_path"].replace("deepsparkhub/", "")
    if is_mmdetection:
        # 选择使用atss作为个例
        prepare_script = f"""
            apt install -y libgl1-mesa-glx
            cd ../cv/detection/atss_mmdet/pytorch/
            cp -r /mnt/deepspark/data/3rd_party/mmdetection-v3.3.0 ./mmdetection > /dev/null 2>&1
            mkdir -p /root/.cache/torch/hub/checkpoints/
            cp /mnt/deepspark/data/checkpoints/resnet50-0676ba61.pth /root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
            mkdir -p mmdetection/data
            ln -s /mnt/deepspark/data/datasets/coco2017 mmdetection/data/coco
            cd mmdetection
            pip install -v -e .
            sed -i 's/python /python3 /g' tools/dist_train.sh
            timeout 1800 bash tools/dist_train.sh configs/atss/atss_r50_fpn_1x_coco.py 4
            """
    elif is_paddledetection:
        # 选择使用CenterNet作为个例
        prepare_script = f"""
            apt install -y libgl1-mesa-glx
            cd ../cv/detection/centernet/paddlepaddle/
            cp -r /mnt/deepspark/data/3rd_party/PaddleDetection-release-2.6 ./PaddleDetection
            cd PaddleDetection
            pip install -r requirements.txt
            python3 setup.py install
            mv dataset/coco dataset/coco-bak
            ln -s /mnt/deepspark/data/datasets/coco2017 dataset/coco
            timeout 1800 python3 tools/train.py -c configs/centernet/centernet_r50_140e_coco.yml --eval
            """
    elif is_yolov:
        # 选择使用yolov5作为个例
        prepare_script = f"""
            cd ../cv/detection/yolov5/pytorch/
            bash ci/prepare.sh
            """
    else:
        prepare_script = f"""
            cd ../{deepsparkhub_path}
            bash ci/prepare.sh
            """

    # add pip list info when in debug mode
    if is_debug():
        pip_list_script = "pip list | grep -E 'numpy|transformer|igie|mmcv|onnx'\n"
        prepare_script = pip_list_script + prepare_script + pip_list_script

    logging.info(f"Start running {model_name} test case")
    r, t = run_script(prepare_script)
    sout = r.stdout
    prec = "fp16"
    pattern = r"Average Precision  \(AP\) @\[ (IoU=0.50[:\d.]*)\s*\| area=   all \| maxDets=\s?\d+\s?\] =\s*([\d.]+)"
    matches = re.findall(pattern, sout)
    epoch_pattern = [r"train: \[", r"Epoch: \[", r".*Epoch\s+gpu_mem", r"total_loss: "]
    combined_pattern = re.compile("|".join(epoch_pattern))
    epoch_matches = bool(combined_pattern.search(sout))
    
    if matches:
        # Get last match and convert to floats
        last_map = map(float, matches[-1])
        print(f"MAP: {last_map}")
        result["result"].setdefault(prec, {"status": "PASS"})
        result["result"][prec]["MAP"] = last_map
    elif epoch_matches:
        result["result"].setdefault(prec, {"status": "PASS"})
        result["result"][prec]["MAP"] = "train timeout"
    else:
        result["result"].setdefault(prec, {"status": "FAIL"})
        print("No match found.")

    result["result"][prec]["Cost time (s)"] = t
    logging.debug(f"matchs:\n{matches}\nepoch_matches:\n{epoch_matches}")
    return result

def run_clf_testcase(model):
    model_name = model["model_name"]
    result = {
        "name": model_name,
        "framework": model["framework"],
        "toolbox": model["toolbox"],
        "category": "cv/classification",
        "result": {},
    }
    is_torchvision = True if model["toolbox"].lower() == "torchvision" else False
    is_mmpretrain = True if model["toolbox"].lower() == "mmpretrain" else False
    is_paddleclas = True if model["toolbox"].lower() == "paddleclas" else False
    is_tf_benchmarks = True if model["toolbox"].lower() == "tensorflow/benchmarks" else False
    deepsparkhub_path = model["deepsparkhub_path"].replace("deepsparkhub/", "")
    dataset_path = "/mnt/deepspark/data/datasets/imagenet"

    # # add pip list info when in debug mode
    # if is_debug():
    #     pip_list_script = "pip list | grep -E 'numpy|transformer|igie|mmcv|onnx'\n"
    #     prepare_script = pip_list_script + prepare_script + pip_list_script

    logging.info(f"Start running {model_name} test case")
    if is_torchvision:
        # 选择使用googlenet作为个例
        prepare_script = f"""
            cd ../{deepsparkhub_path}
            timeout 1800 python3 -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --data-path {dataset_path} --model googlenet --batch-size 512
        """
    elif is_mmpretrain:
        # 选择使用mocov2作为个例
        prepare_script = f"""
            apt install -y libgl1-mesa-glx
            cd ../cv/classification/mocov2/pytorch/
            cp -r /mnt/deepspark/data/3rd_party/mmpretrain-v1.2.0 ./mmpretrain > /dev/null 2>&1
            cd mmpretrain
            python3 setup.py install
            mkdir -p data
            ln -s /mnt/deepspark/data/datasets/imagenet data/
            ln -s /mnt/deepspark/data/checkpoints/mocov2_resnet50_8xb32-coslr-200e_in1k_20220825-b6d23c86.pth ./
            sed -i 's@model = dict(backbone=dict(frozen_stages=4))@model = dict(backbone=dict(frozen_stages=4,init_cfg=dict(type='\''Pretrained'\'', checkpoint='\''./mocov2_resnet50_8xb32-coslr-200e_in1k_20220825-b6d23c86.pth'\'', prefix='\''backbone.'\'')))@' configs/mocov2/benchmarks/resnet50_8xb32-linear-steplr-100e_in1k.py
            timeout 1800 python3 tools/train.py configs/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py
        """
    elif is_paddleclas:
        # 选择使用googlenet作为个例
        prepare_script = f"""
            apt install -y libgl1-mesa-glx
            cd ../cv/classification/googlenet/paddlepaddle/
            cp -r /mnt/deepspark/data/3rd_party/PaddleClas-release-2.6 ./PaddleClas > /dev/null 2>&1
            cd PaddleClas
            pip3 install -r requirements.txt
            python3 setup.py install
            mkdir -p dataset
            ln -s /mnt/deepspark/data/datasets/imagenet dataset/ILSVRC2012
            timeout 1800 python3 -u -m paddle.distributed.launch --gpus=0,1,2,3 tools/train.py -c ppcls/configs/ImageNet/Inception/GoogLeNet.yaml -o Arch.pretrained=False -o Global.device=gpu
        """
    elif is_tf_benchmarks:
        # 选择使用alexnet作为个例
        prepare_script = f"""
            apt install -y libgl1-mesa-glx
            cd ../cv/classification/alexnet/tensorflow/
            ln -s /mnt/deepspark/data/datasets/imagenet_tfrecord/ILSVRC2012 dataset/imagenet_tfrecord
            timeout 1800 bash run_train_alexnet_imagenet.sh
        """
    else:
        prepare_script = f"""
            cd ../{deepsparkhub_path}
            ln -s /mnt/deepspark/data/datasets/imagenet ./
            bash ci/prepare.sh
        """
    r, t = run_script(prepare_script)
    sout = r.stdout
    prec = "fp16"
    pattern = re.compile(r'\* Acc@1 (\d+\.\d+) Acc@5 (\d+\.\d+)')
    matches = pattern.findall(sout)
    epoch_pattern = [r"Epoch: \[", r"Epoch\(train\)   \[", r".*TrainEpoch-", r"INFO Train:.*", r".*\[Train\]\[Epoch.*", r".*images\/sec.*"]
    combined_pattern = re.compile("|".join(epoch_pattern))
    epoch_matches = bool(combined_pattern.search(sout))
    
    if matches:
        # Get last match and convert to floats
        last_acc1, last_acc5 = map(float, matches[-1])
        print(f"Acc@1: {last_acc1}, Acc@5: {last_acc5}")
        result["result"].setdefault(prec, {"status": "PASS"})
        result["result"][prec]["acc1"] = last_acc1
        result["result"][prec]["acc5"] = last_acc5
    elif epoch_matches:
        result["result"].setdefault(prec, {"status": "PASS"})
        result["result"][prec]["acc1"] = "train timeout"
        result["result"][prec]["acc5"] = "train timeout"
    else:
        result["result"].setdefault(prec, {"status": "FAIL"})
        print("No match found.")

    result["result"][prec]["Cost time (s)"] = t
    logging.debug(f"matchs:\n{matches}")
    return result

def run_llm_testcase(model):
    model_name = model["model_name"]
    result = {
        "name": model_name,
        "framework": model["framework"],
        "toolbox": model["toolbox"],
        "category": "cv/classification",
        "result": {},
    }
    is_firefly = True if model["toolbox"].lower() == "firefly" else False
    is_deepspeed = True if model["toolbox"].lower() == "deepspeed" else False
    is_megatron_deepspeed = True if model["toolbox"].lower() == "megatron-deepspeed" else False
    deepsparkhub_path = model["deepsparkhub_path"].replace("deepsparkhub/", "")

    logging.info(f"Start running {model_name} test case")
    if is_firefly:
        # 选择使用qwen-7b作为个例
        # ***** train metrics *****
        # epoch                    =     0.0016
        # total_flos               =  3676485GF
        # train_loss               =     1.1016
        # train_runtime            = 0:02:04.04
        # train_samples_per_second =      3.225
        # train_steps_per_second   =      0.403
        pattern = re.compile(r'^\s*(\S+)\s*=\s*(.+)$', re.MULTILINE)
        prepare_script = f"""
            cd ../toolbox/firefly
            python3 setup.py develop
            cd ../../{deepsparkhub_path}
            mkdir -p data
            ln -s /mnt/deepspark/data/datasets/school_math_0.25M.jsonl data/
            mkdir -p checkpoint
            ln -s /mnt/deepspark/data/checkpoints/qwen-7B checkpoint/
            timeout 1800 bash train.sh 1 configs/qwen-7b-sft-lora.json lora
        """
    elif is_deepspeed:
        # 选择使用chatglm3-6b作为个例
        # {'train_runtime': 84.0969, 'train_samples_per_second': 2.378, 'train_steps_per_second': 1.189, 'train_loss': 0.24943359375, 'epoch': 0.0}
        pattern = r"({.*?})"
        prepare_script = f"""
            cd ../../{deepsparkhub_path}
            pip3 install -r requirements.txt
            mkdir -p data
            ln -s /mnt/deepspark/data/datasets/AdvertiseGen data/
            python3 process_data.py
            mkdir -p checkpoint
            ln -s /mnt/deepspark/data/checkpoints/chatglm3-6b checkpoint/
            timeout 1800 bash run.sh configs/lora.yaml 1
        """
    elif is_megatron_deepspeed:
        # 选择使用llama2-7b作为个例
        prepare_script = f"""
            cd ../toolbox/Megatron-DeepSpeed
            ln -s /mnt/deepspark/data/datasets/gpt_small_117M dataset/
            cd examples/llama2
            sed -i 's/ens5f0/eth0/g' run_llama2_7b_1node.sh
            timeout 1800 bash run_llama2_7b_1node.sh
        """
    else:
        pattern = re.compile(r'^\s*(\S+)\s*=\s*(.+)$', re.MULTILINE)
        prepare_script = f"""
            cd ../{deepsparkhub_path}
            bash ci/prepare.sh
        """
    r, t = run_script(prepare_script)
    sout = r.stdout
    prec = "fp16"
    metrics = {}
    if is_firefly:
        for match in pattern.finditer(sout):
            key = match.group(1).strip()
            value = match.group(2).strip()
            metrics[key] = value
    elif is_deepspeed:
        required_keys = {
            'train_runtime',
            'train_samples_per_second',
            'train_steps_per_second',
            'train_loss',
            'epoch'
        }
        match = re.search(pattern, sout)
        if match:
            dict_str = match.group(1)
            dict_str = dict_str.replace("'", '"')
            try:
                data = json.loads(dict_str)
                if required_keys.issubset(data.keys()):
                    metrics = {key: data[key] for key in required_keys}
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {str(e)}")
    else:
        epoch_pattern = [r".*tokens per second per device.*", r"Epoch: \[", r".*Epoch\s+gpu_mem", r"total_loss: "]
        combined_pattern = re.compile("|".join(epoch_pattern))
        epoch_matches = bool(combined_pattern.search(sout))

    if metrics:
        result["result"].setdefault(prec, {"status": "PASS"})
        for key, value in metrics.items():
            result["result"][prec][key] = value
    elif epoch_matches:
        result["result"].setdefault(prec, {"status": "PASS"})
        result["result"][prec]["tokens per second"] = "train timeout"
    else:
        result["result"].setdefault(prec, {"status": "FAIL"})
        print("No match found.")

    result["result"][prec]["Cost time (s)"] = t
    logging.debug(f"matchs:\n{metrics}")
    return result

def get_metric_result(str):
    if str:
        return json.loads(str.replace("'", "\""))["metricResult"]
    return None

def run_script(script):
    start_time = time.perf_counter()
    result = subprocess.run(
        script, shell=True, capture_output=True, text=True, executable="/bin/bash"
    )
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    logging.debug(f"执行命令：\n{script}")
    logging.debug("执行时间: {:.4f} 秒".format(execution_time))
    logging.debug(f"标准输出: {result.stdout}")
    logging.debug(f"标准错误: {result.stderr}")
    logging.debug(f"返回码: {result.returncode}")
    return result, execution_time

if __name__ == "__main__":
    # 配置日志
    debug_level = logging.DEBUG if is_debug() else logging.INFO
    logging.basicConfig(
        handlers=[logging.FileHandler("output.log"), logging.StreamHandler()],
        level=debug_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    main()
