# Water/se_e2_a
## Model description
The notation of se_e2_a is short for the Deep Potential Smooth Edition (DeepPot-SE) constructed from all information (both angular and radial) of atomic configurations. The e2 stands for the embedding with two-atoms information. This descriptor was described in detail in the DeepPot-SE paper.
Note that it is sometimes called a “two-atom embedding descriptor” which means the input of the embedding net is atomic distances. The descriptor does encode multi-body information (both angular and radial information of neighboring atoms).
In this example, we will train a DeepPot-SE model for a water system. A complete training input script of this example can be found in the directory.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.0     |  23.09  |

## Model Preparation

### Prepare Resources

### Install Dependencies
```
apt install git
git clone --recursive -b v2.2.2 https://github.com/deepmodeling/deepmd-kit.git deepmd-kit
pip3 install numpy==1.22.3
pip3 install deepmd-kit[gpu,cu10,lmp,ipi]==2.2.2
```

#### Install the DeePMD-kit’s Python interface
Visit Iluvatar Corex official website - Resource Center page (https://support.iluvatar.com/#/DocumentCentre?id=1&nameCenter=2&productId=381380977957597184) to obtain the Linux version software stack offline installation package. If you already have an account, click the "Login" button at the upper right corner. If you do not have an account, click the "Login" button and select "Register" to apply for an account. And then download BI_SDK3.1.0 (4.57GB).
```
pip3 install tensorflow-2.6.5+corex.3.1.0-cp38-cp38-linux_x86_64.whl
sed -i '473s/np.float64/np.float32/' deepmd/env.py
pip3 install .
```

#### Install the DeePMD-kit’s C++ interface
```
deepmd_source_dir=`pwd`
cd $deepmd_source_dir/source
mkdir build && cd build
apt install cmake
cmake -DUSE_TF_PYTHON_LIBS=TRUE -DCMAKE_INSTALL_PREFIX=$deepmd_root ..
make -j4
make install
```

#### Install from pre-compiled C library
```
cmake -DDEEPMD_C_ROOT=./libdeepmd_c -DCMAKE_INSTALL_PREFIX=$deepmd_root ..
make -j8
make install
```

#### Install LAMMPS
```
cd $deepmd_source_dir
wget https://github.com/lammps/lammps/archive/stable_23Jun2022_update4.tar.gz
tar xf stable_23Jun2022_update4.tar.gz
mkdir -p lammps-stable_23Jun2022_update4/build/
cd lammps-stable_23Jun2022_update4/build/
apt update
apt install openmpi-bin
apt install libopenmpi-dev
cmake -D PKG_PLUGIN=ON -D PKG_KSPACE=ON -D LAMMPS_INSTALL_RPATH=ON -D BUILD_SHARED_LIBS=yes -D CMAKE_INSTALL_PREFIX=${deepmd_root} -D CMAKE_INSTALL_LIBDIR=lib -D CMAKE_INSTALL_FULL_LIBDIR=${deepmd_root}/lib ../cmake
make -j4
make install
```

#### Install i-PI
```
cd ../..
pip3 install -U i-PI
pip3 install pytest
```

## Model Training

### One single GPU
```
cd $deepmd_source_dir/examples/water/se_e2_a/
export CUDA_VISIBLE_DEVICES=0
export TF_ENABLE_DEPRECATION_WARNINGS=1
DP_INTERFACE_PREC=low dp train input.json
```

## Model Results
| GPU         | average training     |
| ----------- | -------------------- |
| 1 card      | 0.0325 s/batch       |


## References
https://github.com/deepmodeling/deepmd-kit#about-deepmd-kit

