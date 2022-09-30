set -euox pipefail


pip3 install argparse easydict h5py matplotlib numpy open3d==0.10 opencv-python pyyaml scipy tensorboardX timm==0.4.5  tqdm transforms3d termcolor scikit-learn==0.24.1 --default-timeout=1000

# Chamfer Distance
bash install.sh
# PointNet++
cd ./Pointnet2_PyTorch
pip3 install pointnet2_ops_lib/.
cd -
# GPU kNN
pip3 install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

# prepare data

cd data/ShapeNet55-34

if [[ ! -f "ShapeNet55.zip" ]]; then
    wget http://10.150.9.95/swapp/datasets/cv/ShapeNet55.zip
    unzip ShapeNet55.zip
    mv ShapeNet55/shapenet_pc/ .
    rm -r ShapeNet55
fi

cd -

cd ./data/ModelNet/modelnet40_normal_resampled

if [[ ! -f "processed_ModelNet.zip" ]]; then
    wget http://10.150.9.95/swapp/datasets/cv/processed_ModelNet.zip
    unzip processed_ModelNet.zip
    mv processed_ModelNet/* .
fi


apt update
apt install libgl1-mesa-glx



