# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.


######## install packages
yum install -y epel-release
yum install -y jq
pip install -r requirements.txt

######## prepare env
# clean deps/
rm -rf deps/
mkdir -p deps/
# download openmp-13.0.1.src.tar.xz
cd ./deps
wget "https://github.com/llvm/llvm-project/releases/download/llvmorg-13.0.1/openmp-13.0.1.src.tar.xz"
tar -xvJf openmp-13.0.1.src.tar.xz && mv openmp-13.0.1.src openmp
cd openmp/
mkdir build && cd build/

OPENMP_INSTALL_PREFIX=/usr/local/llvmopenmp
cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=${OPENMP_INSTALL_PREFIX} -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=On -DCMAKE_CXX_COMPILER=/opt/rh/devtoolset-7/root/usr/bin/c++ -DCMAKE_C_COMPILER=/opt/rh/devtoolset-7/root/usr/bin/gcc ../
make && make install

cp ${OPENMP_INSTALL_PREFIX}/lib/libomp.so /opt/sw_home/local/lib64/libomp.so
cp ${OPENMP_INSTALL_PREFIX}/include/omp.h /opt/sw_home/local/lib64/clang/13.0.1/include/omp.h

######## install warp-transducer
## back to deps/
cd ../../
git clone https://github.com/HawkAaron/warp-transducer
COMMIT_SHA=f546575109111c455354861a0567c8aa794208a2
cd warp-transducer && git checkout $COMMIT_SHA
mkdir build && cd build

export CUDA_HOME=/opt/sw_home/local/cuda
export CC=/opt/sw_home/local/bin/clang
export CXX=/opt/sw_home/local/bin/clang++
export CUDA_NVCC_EXECUTABLE=/opt/sw_home/local/bin/clang++
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CUDA_HOME/lib64:$LIBRARY_PATH"
export CFLAGS="-I$CUDA_HOME/include $CFLAGS"

cmake -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF ..
make
cd ../pytorch_binding && python3 setup.py install
