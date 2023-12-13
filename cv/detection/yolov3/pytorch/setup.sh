#!/bin/bash
# Install packages
echo "Start installing packages..."
pip3 install tqdm
pip3 install terminaltables
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
if [[ ${ID} == "ubuntu" ]]; then
  echo ${ID}
apt -y install libgl1-mesa-glx
apt -y install libgeos-dev
elif [[ ${ID} == "Loongnix" ]]; then
  echo ${ID}
apt -y install libgl1-mesa-glx
apt -y install libgeos-dev
elif [[ ${ID} == "centos" ]]; then
  echo ${ID}
yum -y install mesa-libGL
yum -y install geos-devel
elif [[ ${ID} == "kylin" ]]; then
  echo ${ID}
yum -y install mesa-libGL
yum -y install geos-devel
else
  echo "Unable to determine OS..."
fi
pip3 install cython # Will automatically install opencv-python
pip3 install imgaug # Will automatically install opencv-python
pip3 install torchsummary

echo "Finished installing packages."

