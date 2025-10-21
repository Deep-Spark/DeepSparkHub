: "${DATA_DIR:="./datasets"}"

if [ ! -d "${DATA_DIR}" ]; then
    mkdir -p "$DATA_DIR"
fi

#download original data STC.json
if [ ! -f "${DATA_DIR}/STC.json" ]; then
    echo "download original data STC.json..."
    gdown -O "${DATA_DIR}/STC.json" --fuzzy https://drive.google.com/uc?id=19VyP6e7pS4pYed87yfvO2hAO7dd0dL9K
fi

#preprocess original data to get train.txt and valid.txt 
if [ ! -f "${DATA_DIR}/train.txt" ] || [ ! -f "${DATA_DIR}/valid.txt" ]; then
    echo "generate train.txt and valid.txt"
python3 ./preprocess_stc_finetune.py --data_dir ${DATA_DIR} --output_dir ${DATA_DIR}
fi

# #download checkpoint QLtmx
# if [ ! -f "${DATA_DIR}/QLtmx" ]; then
#     echo "download checkpoint QLtmx..."
#     # wget -O "${DATA_DIR}/QLtmx" "official url..."
# fi

# #convert checkpoint
# echo "convert checkpoint to MP1..."
# tar -xvf ${DATA_DIR}/QLtmx -C ${DATA_DIR}

#download checkpoint
if [ ! -d "${DATA_DIR}/cpm_model_states_medium.pt" ]; then
    if [ ! -d "${DATA_DIR}/CPM-large" ]; then
        echo "Not find checkpoint CPM-large"
        exit
    fi

    echo "convert checkpoint to MP1..."
    python3 ./change_mp.py ${DATA_DIR}/CPM-large 1
    mv ${DATA_DIR}/CPM-large_MP1/80000/mp_rank_00_model_states.pt ${DATA_DIR}

    echo "convert checkpoint to medium..."
    python3 ./convert_to_medium_from_large.py --load "${DATA_DIR}/mp_rank_00_model_states.pt"

    rm -R ${DATA_DIR}/CPM-large ${DATA_DIR}/CPM-large_MP1 ${DATA_DIR}/._CPM-large
fi