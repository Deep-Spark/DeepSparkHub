set -euox pipefail

GLM_DATA_DIR=$1

set -euox pipefail


if [ ! -n "$GLM_DATA_DIR" ]; then
  echo "set data dir to default"
  GLM_DATA_DIR=/home/data/perf/glm/
fi

echo "data save in "$GLM_DATA_DIR

mkdir -p ${GLM_DATA_DIR}
cd ${GLM_DATA_DIR}

if [[ ! -f "ReCoRD.zip" ]]; then
        echo "ReCoRD.zip not exist"
        echo "wget https://dl.fbaipublicfiles.com/glue/superglue/data/v2/ReCoRD.zip"
        wget https://dl.fbaipublicfiles.com/glue/superglue/data/v2/ReCoRD.zip
fi

unzip -o ReCoRD.zip

cd -

pip3 install -r ./data_preprocessing/requirements.txt

mkdir -p ${GLM_DATA_DIR}/ReCoRD/glm_train_eval_hdf5_sparse/train_hdf5
mkdir -p ${GLM_DATA_DIR}/ReCoRD/glm_train_eval_hdf5_sparse/eval_hdf5

GLM_DATA_DIR=${GLM_DATA_DIR} python3 ./data_preprocessing/create_train_eval_data.py
