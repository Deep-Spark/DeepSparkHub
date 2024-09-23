set -euox pipefail

CUR_DIR=$(cd "$(dirname "$0")";pwd)
cd ${CUR_DIR}

# llama2
if [[ ! -d ${CUR_DIR}/gpt_small_117M ]]; then
    echo "gpt_small dataset not exist, downloading..."
    wget http://10.150.9.95/swapp/datasets/nlp/gpt-2-output-dataset/gpt_small_117M.tar
    tar -xf gpt_small_117M.tar && rm -f gpt_small_117M.tar
fi
