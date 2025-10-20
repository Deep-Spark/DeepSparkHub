: ${EPOCH_ARG:="--num_train_epochs 3"}
: ${BATCH_SIZE:=32}
read py_major py_minor <<< $(python3 -V 2>&1 | awk -F '[ .]' '{print $2, $3}')
if [[ $py_major -eq 3 ]] && (( 9 <= py_minor && py_minor <= 12 )); then
    pip3 install numpy~=1.26.4
else
    pip3 install numpy==1.21.6
fi

DATASET=${ROOT_DIR}/data/datasets/glue/sst2

python3 run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name SST2 \
    --dataset_path ${DATASET} \
    --max_seq_length 128 \
    --batch_size ${BATCH_SIZE}   \
    --learning_rate 2e-5 \
    --logging_steps 100 \
    --save_steps 1000 \
    --output_dir ./tmp/ \
    --device gpu \
    --use_amp False ${EPOCH_ARG} "$@"

exit $?
