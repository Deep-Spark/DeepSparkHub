train_root = 'data/mixture'

train_img_prefix1 = f'{train_root}/SynthText_Add/'
train_ann_file1 = f'{train_root}/SynthText_Add/label.txt'

train1 = dict(
    type='OCRDataset',
    img_prefix=train_img_prefix1,
    ann_file=train_ann_file1,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)

train_list = [train1]