TP=1
PP=8

PROJ_HOME=$(dirname "$PWD")
python3 $PROJ_HOME/tools/checkpoint_util.py \
     --model-type GPT \
     --loader megatron \
     --saver megatron \
     --save-model-type save_huggingface_llama \
     --target-tensor-parallel-size ${TP} \
     --target-pipeline-parallel-size ${PP} \
     --load-dir XXX \
     --save-dir XXX \
     --custom-partition 4 4 4 4 4 4 5 3
