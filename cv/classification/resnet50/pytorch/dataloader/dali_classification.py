# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.


import os

import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, size):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.FileReader(file_root=data_dir, random_shuffle=True)
        self.decode = ops.ImageDecoder(device="cpu", output_type=types.RGB)
        self.res = ops.RandomResizedCrop(device="gpu", size=size, random_area=[0.08, 1.25])
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")

        images = self.decode(self.jpegs)
        images = self.res(images.gpu())
        output = self.cmnp(images)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, size):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.FileReader(file_root=data_dir, random_shuffle=False)
        self.decode = ops.ImageDecoder(device="cpu", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_x=size, resize_y=size)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(size, size),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")

        images = self.decode(self.jpegs)
        images = self.res(images.gpu())
        output = self.cmnp(images)
        return [output, self.labels]


def get_imagenet_iter_dali(type, image_dir, batch_size, num_threads, device_id, size):
    if type == 'train':
        pip_train = HybridTrainPipe(batch_size=batch_size, num_threads=num_threads, device_id=device_id,
                                    data_dir = os.path.join(image_dir, "train"),
                                    size=size)
        pip_train.build()
        dali_iter_train = DALIClassificationIterator(pip_train, size=pip_train.epoch_size("Reader"), last_batch_policy = LastBatchPolicy.DROP)
        return dali_iter_train
    elif type == 'val':
        pip_val = HybridValPipe(batch_size=batch_size, num_threads=num_threads, device_id=device_id,
                                    data_dir = os.path.join(image_dir, "val"),
                                    size=size)
        pip_val.build()
        dali_iter_val = DALIClassificationIterator(pip_val, size=pip_val.epoch_size("Reader"), last_batch_policy = LastBatchPolicy.DROP)
        return dali_iter_val


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='directory to save data to', type=str, default='classification data')
    args = parser.parse_args(arguments)

    train_loader = get_imagenet_iter_dali(type='train', image_dir=args.data_dir,
                                          batch_size=256,
                                          num_threads=4, size=224, device_id=3)

    val_loader = get_imagenet_iter_dali(type="val", image_dir=args.data_dir,
                                          batch_size=256,
                                          num_threads=4, size=224, device_id=3)

    print('start dali train dataloader.')
    start = time.time()
    for epoch in range(20):
        for i, data in enumerate(train_loader):
            images = data[0]["data"].cuda(non_blocking=True)
            labels = data[0]["label"].squeeze().long().cuda(non_blocking=True)

        # WARN: Very important
        train_loader.reset()
        print("Epoch", epoch)
    print('dali iterate time: %fs' % (time.time() - start))
    print('end dali train dataloader.')


    print('start dali val dataloader.')
    start = time.time()
    for i, data in enumerate(val_loader):
        images = data[0]["data"].cuda(non_blocking=True)
        print(images.shape)
        labels = data[0]["label"].squeeze().long().cuda(non_blocking=True)
        print(labels.shape)
    print('dali iterate time: %fs' % (time.time() - start))
    print('end dali val dataloader.')


if __name__ == '__main__':
    import os, time, sys
    import argparse
    sys.exit(main(sys.argv[1:]))