test_root = 'data/mixture'

# test_img_prefix1 = f'{test_root}/IIIT5K/'
test_img_prefix1 = f'{test_root}/icdar_2013/'
test_img_prefix2 = f'{test_root}/icdar_2015/'

# test_ann_file1 = f'{test_root}/IIIT5K/test_label.txt'
test_ann_file1 = f'{test_root}/icdar_2013/test_label_1015.txt'
test_ann_file2 = f'{test_root}/icdar_2015/test_label.txt'

test1 = dict(
    type='OCRDataset',
    img_prefix=test_img_prefix1,
    ann_file=test_ann_file1,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)


test2 = {key: value for key, value in test1.items()}
test2['img_prefix'] = test_img_prefix2
test2['ann_file'] = test_ann_file2

# test3 = {key: value for key, value in test1.items()}
# test3['img_prefix'] = test_img_prefix3
# test3['ann_file'] = test_ann_file3

# test_list = [test1, test2, test3]
test_list = [test1, test2]