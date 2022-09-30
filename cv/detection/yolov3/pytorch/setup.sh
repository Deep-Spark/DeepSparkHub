#!/bin/bash
# Install packages
echo "Start installing packages..."
sudo -E pip3 install tqdm
sudo -E pip3 install terminaltables
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
if [[ ${ID} == "ubuntu" ]]; then
  echo ${ID}
  sudo -E apt -y install libgl1-mesa-glx
  sudo -E apt -y install libgeos-dev
elif [[ ${ID} == "Loongnix" ]]; then
  echo ${ID}
  sudo -E apt -y install libgl1-mesa-glx
  sudo -E apt -y install libgeos-dev
elif [[ ${ID} == "centos" ]]; then
  echo ${ID}
  sudo -E yum -y install mesa-libGL
  sudo -E yum -y install geos-devel
elif [[ ${ID} == "kylin" ]]; then
  echo ${ID}
  sudo -E yum -y install mesa-libGL
  sudo -E yum -y install geos-devel
else
  echo "Unable to determine OS..."
fi
sudo -E pip3 install cython # Will automatically install opencv-python
sudo -E pip3 install imgaug # Will automatically install opencv-python
sudo -E pip3 install torchsummary

echo "Finished installing packages."

