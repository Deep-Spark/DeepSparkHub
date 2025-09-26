CURRENT_DIR=$(cd `dirname $0`; pwd)

ROOT_DIR=${CURRENT_DIR}/../..

cd ${ROOT_DIR}/data/datasets
unzip -q MOT17.zip
mkdir MOT17/images && mkdir MOT17/labels_with_ids
mv ./MOT17/train ./MOT17/images/ && mv ./MOT17/test ./MOT17/images/

cd ${ROOT_DIR}/cv/multi_object_tracking/fairmot/pytorch/
pip3 install Cython
pip3 install -r requirements.txt

python3 src/gen_labels_17.py