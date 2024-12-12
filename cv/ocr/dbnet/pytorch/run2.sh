export PYTHONPATH=$PYTHONPATH:/home/yongle.wu2/learning_tutorial/yongle.wu/DBnet/mmocr
CUDA_VISIBLE_DEVICES=3 python3 tools/train.py configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py
