export NGPUS=4
export CUDA_VISIBLE_DEVICES=4,5,6,7
python3 -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_mlperf.py --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml"  SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 4 SOLVER.MAX_ITER 180000 SOLVER.STEPS "(120000, 160000)" SOLVER.BASE_LR 0.01
