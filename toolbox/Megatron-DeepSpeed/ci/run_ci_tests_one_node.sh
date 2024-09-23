#! /bin/bash

ROOT=$(cd ..; pwd)
cd ${ROOT}


cd tests
bash run_test_one_node.sh
status=$?
if [ $status == 255 ]; then
    exit -1
else
    exit $status
fi