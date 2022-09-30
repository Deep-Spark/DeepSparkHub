export NGPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_mlperf.py --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml"
