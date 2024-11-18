
source get_num_devices.sh
python3 -m torch.distributed.launch --nproc_per_node=$IX_NUM_CUDA_VISIBLE_DEVICES --use_env \
../train.py \
--data-path /home/datasets/cv/imagenet-mini \
--batch-size 256 \
--lr 1e-2 \
--wd 0.0001 \
"$@"
