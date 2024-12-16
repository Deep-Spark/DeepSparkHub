set -euox pipefail

CUR_DIR=$(cd "$(dirname "$0")";pwd)
cd ${CUR_DIR}

# llama2
if [[ ! -d ${CUR_DIR}/gpt_small_117M ]]; then
    echo "gpt_small dataset not exist, downloading..."
    wget http://files.deepspark.org.cn:880/deepspark/gpt_small_117M.tar
    tar -xf gpt_small_117M.tar && rm -f gpt_small_117M.tar
fi
