#!/bin/bash

PYTHON_PATH=$(which python3)

${PYTHON_PATH} setup.py clean || true
rm -rf build build_pip

# Return 0 status if all finished
exit 0
