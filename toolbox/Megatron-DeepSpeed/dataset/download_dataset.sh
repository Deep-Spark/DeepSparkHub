set -euox pipefail

CUR_DIR=$(cd "$(dirname "$0")";pwd)
cd ${CUR_DIR}

if [[ ! -d ${CUR_DIR}/BookCorpusDataset ]]; then
    echo "BookCorpusDataset not exist, downloading..."
    wget http://sw.iluvatar.ai/download/apps/datasets/BookCorpusDataset/BookCorpusDataset.tar
    tar -xf BookCorpusDataset.tar && rm -f BookCorpusDataset.tar
fi