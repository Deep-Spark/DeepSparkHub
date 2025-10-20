#!/bin/bash
PROJPATH=$(dirname $(realpath "$0"))
cd $(dirname $(realpath "$0"))
PYTHON_PATH=$(which python3)

rm -rf build
${PYTHON_PATH} setup.py clean || true
rm -rf build_pip
rm -rf ${PROJPATH}/dist

# Return 0 status if all finished
exit 0