source get_num_devices.sh

export DRT_MEMCPYUSEKERNEL=20000000000
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=$IX_NUM_CUDA_VISIBLE_DEVICES --use_env \
../train.py \
--data-path /home/datasets/cv/VOC2012_sample \
--amp \
--wd 0.000001 \
--lr 0.001 \
--batch-size 4 \
"$@"
