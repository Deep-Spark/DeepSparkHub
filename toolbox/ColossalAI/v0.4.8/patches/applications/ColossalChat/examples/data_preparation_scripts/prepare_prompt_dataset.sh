SAVE_DIR="processed_data/prompt"

rm -rf $SAVE_DIR/cache
rm -rf $SAVE_DIR/jsonl
rm -rf $SAVE_DIR/arrow

input_data_dirs="dataset/competition_math/sft"
conversation_template_config="/home/lin.wu/project/colossalai/applications/ColossalChat/conversation_template/Qwen_Qwen2.5-3B.json"
tokenizer_dir="/home/lin.wu/dataset/Qwen2.5-3B"

python3 prepare_dataset.py --type prompt \
    --input_data_dirs ${input_data_dirs} \
    --conversation_template_config ${conversation_template_config} \
    --tokenizer_dir  ${tokenizer_dir} \
    --data_cache_dir $SAVE_DIR/cache \
    --data_jsonl_output_dir $SAVE_DIR/jsonl \
    --data_arrow_output_dir $SAVE_DIR/arrow \
    --max_length 300
