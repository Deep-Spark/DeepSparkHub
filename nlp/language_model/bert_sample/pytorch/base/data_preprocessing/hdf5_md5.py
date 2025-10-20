import argparse
import h5py
import numpy as np
import hashlib
import os

# Exmaple usage:
# python3 tfrecord_md5sum.py --input_tfrecord=eval_10k --output_md5sum=eval_shard.md5

parser = argparse.ArgumentParser(
    description="HDF5 variable length to MD5sums for BERT.")
parser.add_argument(
    '--input_hdf5',
    type=str,
    required=True,
    help='Input tfrecord path')
args = parser.parse_args()


if __name__ == '__main__':

  h = hashlib.md5

row_sums=[]
f = h5py.File(args.input_hdf5, 'r')
for i in range(f['input_ids'].shape[0]):
    row_sums.append(h(str(f['input_ids'][i].tolist()).encode('utf-8')).hexdigest())
f.close()
print("{}\t{}".format(os.path.basename(args.input_hdf5), h(str(row_sums).encode('utf-8')).hexdigest()))