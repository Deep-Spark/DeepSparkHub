#1gpu
python3 -m torch.distributed.launch --nproc_per_node=1 --use_env \
    main_nerf.py data/fox --workspace trial_nerf  --num_gpus=1
#8gpus
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=8 --use_env \
#    main_nerf.py data/fox --workspace trial_nerf  --num_gpus=8

