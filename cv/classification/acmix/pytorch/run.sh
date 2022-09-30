NGPU=$1
CONFIG=$2
DATA_PATH=$3

cd Swin-Transformer

python3 -m torch.distributed.launch --nproc_per_node ${NGPU} \
  --master_port 12345 main.py --cfg configs/${CONFIG} \
  --data-path ${DATA_PATH} --batch-size 128
