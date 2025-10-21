#!/bin/bash

PYTHON_PATH=$(which pip3)

rm -rf build
${PYTHON_PATH} uninstall satrn-full
rm -rf build_pip
rm -rf satrn*.egg-info
rm -f compile.log

# Return 0 status if all finished
exit 0
