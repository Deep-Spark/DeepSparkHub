#!/bin/bash
ROOT_DIR="$(cd "$(dirname "$0")/../.."; pwd)"
SRC_DIR=$ROOT_DIR/../audio/speech_recognition/conformer/pytorch
DATA_DIR=$ROOT_DIR/data

# determine whether the user is root mode to execute this script
prefix_sudo=""
current_user=$(whoami)
if [ "$current_user" != "root" ]; then
    echo "User $current_user need to add sudo permission keywords"
    prefix_sudo="sudo"
fi

echo "prefix_sudo= $prefix_sudo"

ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
if [[ ${ID} == "ubuntu" ]]; then
    $prefix_sudo apt install -y numactl
    $prefix_sudo apt install -y libsndfile1
elif [[ ${ID} == "centos" ]]; then
    $prefix_sudo yum install -y numactl
    $prefix_sudo yum install -y libsndfile-devel
else
    echo "Unable to determine OS, assumed to be similar to CentOS"
    $prefix_sudo yum install -y numactl
    $prefix_sudo yum install -y libsndfile-devel
fi

pip3 list | grep "torchaudio" || \
    pip3 install torchaudio==0.8.1

pip3 install --no-index --find-links=$DATA_DIR/packages IXPyLogger==1.0.0
pip3 install -r $SRC_DIR/requirements.txt --cache-dir=$DATA_DIR/packages
pip3 install numpy==1.26.4
if [ ! -f "${HOME}/.cache/librosa/admiralbob77_-_Choice_-_Drum-bass.ogg" ]; then
    wget https://librosa.org/data/audio/admiralbob77_-_Choice_-_Drum-bass.ogg
    mkdir -p ~/.cache/librosa/
    mv admiralbob77_-_Choice_-_Drum-bass.ogg ~/.cache/librosa/
fi

DATASET_DIR=$DATA_DIR/datasets/LibriSpeech

cd $DATASET_DIR
GZS=`find . -type f -name '*.gz'`
for path in $GZS; do
    cd ${path%/*}
    tar zxf ${path##*/}
done

mv $DATASET_DIR/LibriSpeech/* $DATASET_DIR/
