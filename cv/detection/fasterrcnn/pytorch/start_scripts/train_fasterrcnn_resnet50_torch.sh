
CUDA_VISIBLE_DEVICES=0 python3 ../train.py \
--data-path /home/datasets/cv/VOC2012_sample \
--lr 0.001 \
--batch-size 4 \
"$@"
