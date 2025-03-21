set -euox pipefail

CUR_DIR=$(cd "$(dirname "$0")";pwd)
cd ${CUR_DIR}

## llama3
if [[ ! -d ${CUR_DIR}/gpt_small_117M_llama3 ]]; then
    echo "gpt_small dataset not exist, downloading..."
    wget http://10.150.9.95/swapp/datasets/nlp/gpt-2-output-dataset/gpt_small_117M_llama3.tar
    tar -xf gpt_small_117M_llama3.tar && rm -f gpt_small_117M_llama3.tar
fi