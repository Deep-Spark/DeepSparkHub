#!/bin/bash

PYTHON_PATH=$(which python3)

rm -rf build
${PYTHON_PATH} setup.py clean || true
rm -rf build_pip
rm -rf ipex.egg-info
rm -rf deepspeed/git_version_info_installed.py
# Return 0 status if all finished
exit 0
