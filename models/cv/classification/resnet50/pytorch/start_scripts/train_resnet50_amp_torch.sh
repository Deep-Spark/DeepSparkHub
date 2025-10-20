
python3 ../train.py \
--data-path /home/datasets/cv/imagenet-mini \
--batch-size 256 \
--lr 0.01 \
--amp \
"$@"
