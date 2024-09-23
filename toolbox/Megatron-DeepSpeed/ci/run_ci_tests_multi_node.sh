#! /bin/bash

ROOT=$(cd ..; pwd)
cd ${ROOT}


cd tests
bash run_test_multi_node.sh
## 获取退出码
status=$(cat exit_code.txt)

if [[ $status == 255 ]]; then
    exit -1
else
    exit $status
fi