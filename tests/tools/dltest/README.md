### 1. Install dltest tool
    
    python setup.py develop

### 2. Usage

#### 2.1 Fetch log

Commmand:

```shell
ixdltest-fetch args ${log_path}
```

Arguments:

- p or patterns, The pattern of fetch log;
- pn or pattern_names, The name of pattern;
- use_re, Whether use regular expression;
- d or nearest_distance, default=10, The nearest distance of matched pattern;
- start_flag, The flag of start to record log;
- end_flag, The flag of stop to record log;
- split_pattern, The pattern is used to match line, If the line is matched, argument `split_sep` to split the line.
- split_sep, The seperator is used to split line;
- split_idx, The index of split line;
- saved, Save result to path;
- log, Log path.

> Examples

2.1.1. Fetch accuracy in the log of ResNet by match.

```shell
ixdltest-fetch nv-train_resnet50_torch.sh.epoch_5.log -p "Acc@1" "Acc@5"

# Output:
# {'results': [{'Acc@1': [9.682], 'Acc@5': [50.293]}, {'Acc@1': [19.541], 'Acc@5': [61.096]},
#              {'Acc@1': [21.35], 'Acc@5': [67.338]}, {'Acc@1': [21.197], 'Acc@5': [67.083]},
#              {'Acc@1': [24.586], 'Acc@5': [67.949]}]}
```

2.1.2. Fetch mAP in the log of YoloV5 by split.

```shell
ixdltest-fetch nv-train_yolov5s_coco_torch.sh.epoch_5.log \ 
-p "Average Precision  \(AP\) @\[ IoU=0.50:0.95 \| area=   all \| maxDets=100 \] =" \
-pn "mAP"

# Output:
# {'results': [{'mAP': [0.359]}, {'mAP': [0.359]}, {'mAP': [0.359]}]}
```


#### 2.2. Compare logs

```shell
ixdltest-compare --log1 ${log_path1} --log2 ${log_path2} args
```

Arguments:

- log1, First log;
- log2, Second log;
- threshold, Threshold;
- only_last, Whether use the last result to compare;
- print_result, Whether print result;
- p or patterns, The pattern of fetch log;
- pn or pattern_names, The name of pattern;
- use_re, Whether use regular expression;
- d or nearest_distance, default=10, The nearest distance of matched pattern;
- start_flag, The flag of start to record log;
- end_flag, The flag of stop to record log;
- split_pattern, The pattern is used to match line, If the line is matched, argument `split_sep` to split the line.
- split_sep, The seperator is used to split line;
- split_idx, The index of split line;
- saved, Save result to path;
- log, Log path.

> Examples

2.2.1. Compare log

```shell
ixdltest-compare \
--log1 nv-train_resnet50_torch.sh.epoch_5.log \
--log2 nv-train_resnet50_torch.sh.epoch_default.log -p "Acc@1" \
--threshold 0.02

# Output:
# Fail
```

#### 2.3. Validate model

```shell
ixdltest-validate args ${script} ${script_args}
```

Arguments:

- l or compare_log, If None is given, a comparable log will be searched in `deeplearningsamples/runing_logs`;
- saved, Save result to path;
- with_exit_code, Add exit code for the result of compared;
- print_result, Whether print result;
- capture_output, optional values ['pipe', 'tempfile'], The method of capture output;
- run_script, A runable script with arguments.


> Examples

2.3.1. Validate model

```shell
ixdltest-validate bash train_shufflenetv2_x0_5_torch.sh --epochs 5

# Output:
# SUCCESS

```


