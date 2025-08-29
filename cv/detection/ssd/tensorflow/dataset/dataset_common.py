from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}

data_splits_num = {
    'train': 22136,
    'val': 4952,
}

def tf1_get_batch(num_classes, batch_size, split_name, file_pattern, num_readers, num_preprocessing_threads, image_preprocessing_fn, anchor_encoder, num_epochs=None, is_training=True):
    if split_name not in data_splits_num:
        raise ValueError('split name %s was not recognized.' % split_name)

    def parse_tfrecord(example_proto):
        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/shape': tf.FixedLenFeature([3], tf.int64),
            'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
            'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
        }
        parsed_features = tf.parse_single_example(example_proto, keys_to_features)

        image = tf.image.decode_jpeg(parsed_features['image/encoded'], channels=3)
        shape = parsed_features['image/shape']
        filename = parsed_features['image/filename']

        xmin = tf.sparse.to_dense(parsed_features['image/object/bbox/xmin'])
        ymin = tf.sparse.to_dense(parsed_features['image/object/bbox/ymin'])
        xmax = tf.sparse.to_dense(parsed_features['image/object/bbox/xmax'])
        ymax = tf.sparse.to_dense(parsed_features['image/object/bbox/ymax'])
        gbboxes_raw = tf.stack([ymin, xmin, ymax, xmax], axis=1)
        glabels_raw = tf.sparse.to_dense(parsed_features['image/object/bbox/label'])
        isdifficult = tf.sparse.to_dense(parsed_features['image/object/bbox/difficult'])

        return image, filename, shape, glabels_raw, gbboxes_raw, isdifficult

    # 构建 tf.data 管道
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=is_training)  # 支持通配符
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=num_readers),
        cycle_length=num_readers,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=num_preprocessing_threads)
    
    total_samples = data_splits_num[split_name]
    dataset = dataset.take(total_samples)
    print(f"split_name:{split_name}, batch_size:{batch_size}, num_epochs:{num_epochs}, samples_per_epoch: {total_samples},  batch_nums_per_epoch:{total_samples // batch_size}")

    if num_epochs is not None:
        dataset = dataset.repeat(num_epochs)
    else:
        dataset = dataset.repeat()  # 无限重复

    if is_training:
        dataset = dataset.map(lambda img, fn, sh, lbl, box, diff: (
            img, fn, sh, tf.cond(
                tf.count_nonzero(diff, dtype=tf.int32) < tf.shape(diff)[0],
                lambda: tf.boolean_mask(lbl, diff < tf.ones_like(diff)),
                lambda: tf.boolean_mask(lbl, tf.one_hot(0, tf.shape(diff)[0], on_value=True, off_value=False, dtype=tf.bool))
            ),
            tf.cond(
                tf.count_nonzero(diff, dtype=tf.int32) < tf.shape(diff)[0],
                lambda: tf.boolean_mask(box, diff < tf.ones_like(diff)),
                lambda: tf.boolean_mask(box, tf.one_hot(0, tf.shape(diff)[0], on_value=True, off_value=False, dtype=tf.bool))
            )
        ), num_parallel_calls=num_preprocessing_threads)

    # 预处理
    def preprocess(image, filename, shape, glabels_raw, gbboxes_raw):
        if is_training:
            image, glabels, gbboxes = image_preprocessing_fn(image, glabels_raw, gbboxes_raw)
        else:
            image = image_preprocessing_fn(image, glabels_raw, gbboxes_raw)
            glabels, gbboxes = glabels_raw, gbboxes_raw
        return image, filename, shape, glabels, gbboxes

    dataset = dataset.map(preprocess, num_parallel_calls=num_preprocessing_threads)

    # anchor 编码
    def encode(glabels, gbboxes):
        gt_targets, gt_labels, gt_scores = anchor_encoder(glabels, gbboxes)
        return gt_targets, gt_labels, gt_scores

    dataset = dataset.map(lambda img, fn, sh, lbl, box: (
        img, fn, sh, *encode(lbl, box)
    ), num_parallel_calls=num_preprocessing_threads)

    # 批处理
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # 创建 iterator
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()  # 返回 (image, filename, shape, gt_targets, gt_labels, gt_scores)