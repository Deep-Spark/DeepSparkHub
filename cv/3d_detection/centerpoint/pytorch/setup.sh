# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.

cd det3d/ops/dcn 
python3 setup.py build_ext --inplace

cd .. && cd  iou3d_nms
python3 setup.py build_ext --inplace
