#/bin/bash
TP=4
PP=4

PROJ_HOME=$(dirname "$PWD")

## llama2-7B
python3 $PROJ_HOME/tools/checkpoint_util.py \
     --model-type GPT \
     --loader llama2_hf \
     --saver megatron \
     --target-tensor-parallel-size ${TP} \
     --target-pipeline-parallel-size ${PP} \
     --load-dir ./output_step1_llama2_7b \
     --save-dir ./rlhf_llama2_7b_tp${TP}_pp${PP} \
     --tokenizer-model ./output_step1_llama2_7b/tokenizer.model

## tinyllama-1.1B
python3 $PROJ_HOME/tools/checkpoint_util.py \
     --model-type GPT \
     --loader tinyllama_rlhf \
     --saver megatron \
     --target-tensor-parallel-size ${TP} \
     --target-pipeline-parallel-size ${PP} \
     --load-dir ./output_tinyLlama-1.1B-intermediate-step-240k-503b \
     --save-dir ./rlhf_tinyllama_1.1b_tp${TP}_pp${PP} \
     --tokenizer-model ./output_tinyLlama-1.1B-intermediate-step-240k-503b/tokenizer.model \
     --tinyllama \
     --custom-partition 5 5 6 6

mv ./rlhf_llama2_7b_tp${TP}_pp${PP}/iter_0000001/* ./rlhf_llama2_7b_tp${TP}_pp${PP}
mv ./rlhf_tinyllama_1.1b_tp${TP}_pp${PP}/iter_0000001/* ./rlhf_tinyllama_1.1b_tp${TP}_pp${PP}
