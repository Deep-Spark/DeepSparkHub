#!/bin/bash
ROOT_DIR="$(cd "$(dirname "$0")/../.."; pwd)"
SRC_DIR=$ROOT_DIR/../models/multimodal/diffusion_model/stable-diffusion-2.1/pytorch
DATA_DIR=$ROOT_DIR/data

# install packages
pip3 install --no-index --find-links=$DATA_DIR/packages IXPyLogger==1.0.0
pip3 install --upgrade pillow
pip3 install huggingface_hub==0.25.1
pip3 install $DATA_DIR/packages/addons/transformers-4.38.1-py3-none-any.whl
pip3 install -r $SRC_DIR/examples/text_to_image/requirements.txt --cache-dir=$DATA_DIR/packages
bash $SRC_DIR/build_diffusers.sh && bash $SRC_DIR/install_diffusers.sh

# unzip dataset and checkpoints
if [[ ! -d "$DATA_DIR/datasets/pokemon-blip-captions" ]]; then
    echo "Unarchive pokemon-blip-captions.tar"
    tar -xvf $DATA_DIR/datasets/pokemon-blip-captions.tar -C $DATA_DIR/datasets
fi
if [[ ! -d "$DATA_DIR/model_zoo/stabilityai" ]]; then
    echo "Unarchive stabilityai.tar"
    tar -xvf $DATA_DIR/model_zoo/stabilityai.tar -C $DATA_DIR/model_zoo
fi
if [[ ! -d "$DATA_DIR/model_zoo/stable-diffusion-v1-5" ]]; then
    echo "Unarchive stable-diffusion-v1-5.zip"
    unzip $DATA_DIR/model_zoo/stable-diffusion-v1-5.zip -d $DATA_DIR/model_zoo
fi 
