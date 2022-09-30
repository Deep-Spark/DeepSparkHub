
CUDA_VISIBLE_DEVICES=0 python3 ../train.py \
--data-path /home/datasets/cv/coco \
--dataset coco \
--lr 0.001 \
--batch-size 4 \
"$@"
