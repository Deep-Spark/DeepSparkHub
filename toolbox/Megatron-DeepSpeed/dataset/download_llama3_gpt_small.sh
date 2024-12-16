set -euox pipefail

CUR_DIR=$(cd "$(dirname "$0")";pwd)
cd ${CUR_DIR}

## llama3
if [[ ! -d ${CUR_DIR}/gpt_small_117M_llama3 ]]; then
    echo "gpt_small dataset not exist, downloading..."
    wget http://files.deepspark.org.cn:880/deepspark/gpt_small_117M_llama3.tar
    tar -xf gpt_small_117M_llama3.tar && rm -f gpt_small_117M_llama3.tar
fi