# coding=utf-8

# create train and eval data, saved to h5py files

import os
import sys
import h5sparse as h5py
import numpy as np
from tqdm import trange
import pickle
import copy
import scipy.sparse as ss
import os

from dataset import RecordDateset
from utils.tokenization import prepare_tokenizer


class basic_config:
    base_dir = os.environ.get("GLM_DATA_DIR",None)
    if not base_dir:
        print("not found GLM_DATA_DIR in os.environ")
        exit(1)
    raw_data_dir = os.path.join(base_dir,'ReCoRD')
    max_seq_length = 512
    train_data_dir = os.path.join(raw_data_dir,'glm_train_eval_hdf5_sparse/train_hdf5')
    eval_data_dir = os.path.join(raw_data_dir,'glm_train_eval_hdf5_sparse/eval_hdf5')

def create_h5py_file_train(dataset, start_idx, end_idx, save_dir):
    # 将dataset中指定的数据写入h5py文件中
    '''
    text: c,s
    position: c,2,s
    mask: c
    target: c,s
    logit_mask: c,s
    '''
    # 建立两个索引，保存choice维度的索引与answer_idx维度的索引
    data = {'text': [], 'position': [],
            'mask': [], 'target': [], 'logit_mask': [],
            'choice_start_end': []}
    print(
        f"start idx: {start_idx}, end_idx: {end_idx}, dataset length:{len(dataset)}")

    choice_start = 0

    for i in trange(end_idx-start_idx):
        idx = i + start_idx
        sample = dataset[idx]

        choice_len = len(sample['text'])

        for key in sample:
            if key in data:
                data[key].append(sample[key])
        data['choice_start_end'].append(
            [choice_start, choice_start+choice_len])

        choice_start += choice_len

    save_file = os.path.join(save_dir, f'train_sparse.hdf5')
    with h5py.File(save_file, 'w') as f:
        for key in data:
            if key in ['choice_start_end']:
                np_data = np.array(data[key])
            else:
                np_data = np.concatenate(data[key], axis=0)
            print(key, np_data.shape)
            if key == 'position':
                np_data = np_data.reshape([-1, np_data.shape[-1]])
            elif len(np_data.shape) == 1:
                np_data = np_data.reshape([-1, 1])
            ss_data = ss.csc_matrix(np_data)
            f[key] = ss_data


def create_h5py_file_eval(dataset, start_idx, end_idx, save_dir):
    # 将dataset中指定的数据写入h5py文件中
    '''
    text: c,s
    answer_idx: n 不确定
    position: c,2,s
    mask: c
    target: c,s
    logit_mask: c,s
    '''
    # 建立两个索引，保存choice维度的索引与answer_idx维度的索引
    data = {'text': [], 'answer_idx': [], 'position': [],
            'mask': [], 'target': [], 'logit_mask': [],
            'choice_start_end': [], 'answer_start_end': []}
    print(
        f"start idx: {start_idx}, end_idx: {end_idx}, dataset length:{len(dataset)}")

    choice_start = 0
    answer_start = 0

    for i in trange(end_idx-start_idx):
        idx = i + start_idx
        sample = dataset[idx]

        choice_len = len(sample['text'])
        answer_len = len(sample['answer_idx'])

        for key in sample:
            data[key].append(sample[key])
        data['choice_start_end'].append(
            [choice_start, choice_start+choice_len])
        data['answer_start_end'].append(
            [answer_start, answer_len+answer_start])

        choice_start += choice_len
        answer_start += answer_len

    save_file = os.path.join(save_dir, f'eval_sparse.hdf5')
    with h5py.File(save_file, 'w') as f:
        for key in data:
            if key in ['choice_start_end', 'answer_start_end']:
                np_data = np.array(data[key])
            else:
                np_data = np.concatenate(data[key], axis=0)
            print(key, np_data.shape)
            if key == 'position':
                np_data = np_data.reshape([-1, np_data.shape[-1]])
            elif len(np_data.shape) == 1:
                np_data = np_data.reshape([-1, 1])
            ss_data = ss.csc_matrix(np_data)
            f[key] = ss_data


def main():
    config = basic_config()
    tokenizer = prepare_tokenizer()
    train_dataset = RecordDateset(config, "train", tokenizer)
    eval_dataset = RecordDateset(config, "dev", tokenizer)

    # max_choice_num, max_answer_idx = compute_max_length(eval_dataset)
    # print(max_choice_num, max_answer_idx)

    train_choice_num = 10
    train_answer_num = 1

    if not os.path.isdir(config.eval_data_dir):
        print(f"create eval data dir:{config.eval_data_dir}")
        os.makedirs(config.eval_data_dir)
    if not os.path.isdir(config.train_data_dir):
        print(f"create train data dir:{config.train_data_dir}")
        os.makedirs(config.train_data_dir)

    print("============save file for eval============")
    create_h5py_file_eval(eval_dataset, 0, len(eval_dataset),
                          config.eval_data_dir)

    print("============save file for train============")
    train_length = len(train_dataset)
    start_idx = 0
    create_h5py_file_train(train_dataset,start_idx,train_length,config.train_data_dir)


if __name__ == "__main__":
    main()
