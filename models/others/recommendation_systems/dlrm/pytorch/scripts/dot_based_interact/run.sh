#!/bin/bash

python "$(dirname "$(realpath "$0")")"/baseline.py --batch_size="$@"
python "$(dirname "$(realpath "$0")")"/custom_kernel.py --batch_size="$@"