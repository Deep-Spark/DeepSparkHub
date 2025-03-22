set -euox pipefail

CUR_DIR=$(cd "$(dirname "$0")";pwd)
cd ${CUR_DIR}

if [[ ! -d ${CUR_DIR}/RedPajama-Data-1T-Sample ]]; then
    echo "RedPajama-Data-1T-Sample dataset not exist, downloading..."
    wget http://sw.iluvatar.ai/download/apps/datasets/nlp/RedPajama-Data-1T-Sample/RedPajama-Data-1T-Sample.tar
    tar -xf RedPajama-Data-1T-Sample.tar && rm -f RedPajama-Data-1T-Sample.tar
fi