export PYTORCH_DISABLE_VEC_KERNEL=1
export PT_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT=1
CUDA_VISIBLE_DEVICES=0 python3 ../train.py \
--data-path /home/datasets/cv/VOC2012_sample \
--amp \
--lr 0.001 \
--batch-size 1  \
"$@"
