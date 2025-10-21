# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: generate inputs and targets for the dlrm benchmark
# The inpts and outputs are generated according to the following three option(s)
# 1) random distribution
# 2) synthetic distribution, based on unique accesses and distances between them
#    i) R. Hassan, A. Harris, N. Topham and A. Efthymiou "Synthetic Trace-Driven
#    Simulation of Cache Memory", IEEE AINAM'07
# 3) public data set
#    i)  Criteo Kaggle Display Advertising Challenge Dataset
#    https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset
#    ii) Criteo Terabyte Dataset
#    https://labs.criteo.com/2013/12/download-terabyte-click-logs


# others
from os import path
import data_utils
import data_loader_terabyte
import argparse
import dataset
import numpy as np
import os

def CriteoDataset(
        dataset,
        max_ind_range,
        sub_sample_rate,
        randomize,
        split="train",
        raw_path="",
        pro_data="",
        memory_map=False,
        dataset_multiprocessing=False,
        day_num=24
):
    # dataset
    # tar_fea = 1   # single target
    den_fea = 13  # 13 dense  features
    # spa_fea = 26  # 26 sparse features
    # tad_fea = tar_fea + den_fea
    # tot_fea = tad_fea + spa_fea

    days = day_num
    out_file = "terabyte_processed"

    # split the datafile into path and filename
    lstr = raw_path.split("/")
    d_path = "/".join(lstr[0:-1]) + "/"
    d_file = lstr[-1]
    npzfile = d_path +  d_file

    # check if pre-processed data is available
    data_ready = True
    if memory_map:
        for i in range(days):
            reo_data = npzfile + "_{0}_reordered.npz".format(i)
            if not path.exists(str(reo_data)):
                data_ready = False
    else:
        if not path.exists(str(pro_data)):
            data_ready = False

    # pre-process data if needed, generate ”.npz” files
    # WARNNING: when memory mapping is used we get a collection of files
    if data_ready:
        print("Reading pre-processed data=%s" % (str(pro_data)))
        file = str(pro_data)
    else:
        print("Reading raw data=%s" % (str(raw_path)))
        file = data_utils.getCriteoAdData(
            raw_path,
            out_file,
            max_ind_range,
            sub_sample_rate,
            days,
            split,
            randomize,
            criteo_kaggle=False,
            memory_map=memory_map,
            dataset_multiprocessing=dataset_multiprocessing,
        )

def ensure_dataset_preprocessed(args, day_num):
    _ = CriteoDataset(
        args.data_set,
        args.max_ind_range,
        args.data_sub_sample_rate,
        args.data_randomize,
        "train",
        args.raw_data_file,
        args.processed_data_file,
        args.memory_map,
        args.dataset_multiprocessing,
        day_num
    )

    _ = CriteoDataset(
        args.data_set,
        args.max_ind_range,
        args.data_sub_sample_rate,
        args.data_randomize,
        "test",
        args.raw_data_file,
        args.processed_data_file,
        args.memory_map,
        args.dataset_multiprocessing,
        day_num
    )

    for split in ['train', 'val', 'test']:
        print('Running preprocessing for split =', split)

        train_files = ['{}_{}_reordered.npz'.format(args.raw_data_file, day)
                       for day in range(0, day_num - 1)]

        test_valid_file = args.raw_data_file + '_{}_reordered.npz'.format(day_num-1)

        args.processed_data_file.split('.')[0]
        output_file = args.processed_data_file.split('.')[0] + '_{}.bin'.format(split)

        input_files = train_files if split == 'train' else [test_valid_file]
        data_loader_terabyte.numpy_to_binary(input_files=input_files,
                                             output_file_path=output_file,
                                             split=split)
def extract_file(args): 
    days = args.extract_days.split(',')
    for day in days:
        raw_data_path = args.raw_data_file+'_' + day
        extract_data_path = raw_data_path+'_ext'

        raw_line_nums = 0
        ext_line_nums = 0
        with open(str(raw_data_path)) as f:
            for _ in f:
                raw_line_nums += 1
        print(f"raw file:{raw_data_path} line_nums:{raw_line_nums}")

        rand_u = np.random.uniform(low=0.0, high=1.0, size=raw_line_nums)
        f_raw = open(raw_data_path, 'rb')
        if os.path.exists(extract_data_path):
            os.system("rm "+extract_data_path)
        f_ext = open(extract_data_path, 'ab')
        for i in range(raw_line_nums):
            line = f_raw.readline()
            if rand_u[i] > args.extract_sample_rate:
                continue
            else:
                f_ext.write(line)
                ext_line_nums+=1

        print(f"ext file:{extract_data_path} line_nums:{ext_line_nums}")

        f_raw.close()
        f_ext.close()

if __name__ == "__main__":

    ### parse arguments ###
    parser = argparse.ArgumentParser(description="Generate datasets")

    parser.add_argument("--fun", type=str, default="") 
    parser.add_argument("--data-set", type=str, default="terabyte") 
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--memory-map", action="store_true", default=False)
    parser.add_argument(
        "--dataset-multiprocessing",
        action="store_true",
        default=False,
        help="The Kaggle dataset can be multiprocessed in an environment \
                        with more than 7 CPU cores and more than 20 GB of memory. \n \
                        The Terabyte dataset can be multiprocessed in an environment \
                        with more than 24 CPU cores and at least 1 TB of memory.",
    )
    parser.add_argument("--day-num", type=int, default=-1)

    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--test_batch_size", type=int, default=100)
    parser.add_argument("--extract-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--extract-days", type=str, default='')

    args = parser.parse_args()

    np.random.seed(1234)

    if args.fun == 'preprocess':
        ensure_dataset_preprocessed(args, args.day_num)
    elif args.fun == 'query':
        data_loader_train, data_loader_test = dataset.get_data_loader(
            args.dataset, args.batch_size, args.test_batch_size)
        print(f"data_loader_train length: {len(data_loader_train)}")
        print(f"data_loader_test length: {len(data_loader_test)}")
    elif args.fun == "extract":
        extract_file(args)
