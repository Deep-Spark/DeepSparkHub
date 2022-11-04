#!/bin/bash

echo "prepare data..."

git clone https://github.com/neheller/kits19
cd kits19
pip3 install -r requirements.txt
python3 -m starter_code.get_imaging
cd ..

mkdir -p data/kits19/train

python3 preprocess_dataset.py --data_dir kits19/data --results_dir data/kits19/train

echo "data done!"

