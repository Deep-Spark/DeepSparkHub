from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from net import ssd_net

from dataset import dataset_common
from preprocessing import ssd_preprocessing
from utility import anchor_manipulator
from utility import scaffolds

# hardware related configuration
tf.app.flags.DEFINE_integer(
    'num_readers', 8,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 24,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer(
    'num_cpu_threads', 0,
    'The number of cpu cores used to train.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 1., 'GPU memory fraction to use.')
# scaffold related configuration
tf.app.flags.DEFINE_string(
    'data_dir', './dataset/tfrecords',
    'The directory where the dataset input data is stored.')
tf.app.flags.DEFINE_integer(
    'num_classes', 21, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'model_dir', './logs/',
    'The directory where the model will be stored.')
tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are printed.')
tf.app.flags.DEFINE_integer(
    'save_summary_steps', 500,
    'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_checkpoints_secs', 7200,
    'The frequency with which the model is saved, in seconds.')
# model related configuration
tf.app.flags.DEFINE_integer(
    'train_image_size', 300,
    'The size of the input image for the model to use.')
tf.app.flags.DEFINE_integer(
    'train_epochs', 5,
    'The number of epochs to use for training.')
tf.app.flags.DEFINE_integer(
    'max_number_of_steps', 120000,
    'The max number of steps to use for training.')
tf.app.flags.DEFINE_integer(
    'batch_size', 32,
    'Batch size for training and evaluation.')
tf.app.flags.DEFINE_string(
    'data_format', 'channels_first', # 'channels_first' or 'channels_last'
    'A flag to override the data format used in the model. channels_first '
    'provides a performance boost on GPU but is not always compatible '
    'with CPU. If left unspecified, the data format will be chosen '
    'automatically based on whether TensorFlow was built for CPU or GPU.')
tf.app.flags.DEFINE_float(
    'negative_ratio', 3., 'Negative ratio in the loss function.')
tf.app.flags.DEFINE_float(
    'match_threshold', 0.5, 'Matching threshold in the loss function.')
tf.app.flags.DEFINE_float(
    'neg_threshold', 0.5, 'Matching threshold for the negtive examples in the loss function.')
# optimizer related configuration
tf.app.flags.DEFINE_integer(
    'tf_random_seed', 20180503, 'Random seed for TensorFlow initializers.')
tf.app.flags.DEFINE_float(
    'weight_decay', 5e-4, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.000001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
# for learning rate piecewise_constant decay
tf.app.flags.DEFINE_string(
    'decay_boundaries', '500, 80000, 100000',
    'Learning rate decay boundaries by global_step (comma-separated list).')
tf.app.flags.DEFINE_string(
    'lr_decay_factors', '0.1, 1, 0.1, 0.01',
    'The values of learning_rate decay factor for each segment between boundaries (comma-separated list).')
# checkpoint related configuration
tf.app.flags.DEFINE_string(
    'checkpoint_path', './model',
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'checkpoint_model_scope', 'vgg_16',
    'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
    'model_scope', 'ssd300',
    'Model scope name used to replace the name_scope in checkpoint.')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', 'ssd300/multibox_head, ssd300/additional_layers, ssd300/conv4_3_scale',
    'Comma-separated list of scopes of variables to exclude when restoring from a checkpoint.')
tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', True,
    'When restoring a checkpoint would ignore missing variables.')
tf.app.flags.DEFINE_boolean(
    'multi_gpu', True,
    'Whether there is GPU to use for training.')
tf.app.flags.DEFINE_boolean(
    'use_amp', False,
    'Whether to use amp for training.')
tf.app.flags.DEFINE_string(
    'backbone', 'vgg16',
    'The backbone for feature extraction: vgg16/resnet18/resnet34/resnet50/resnet101.')

FLAGS = tf.app.flags.FLAGS

def get_init_fn():
    return scaffolds.get_init_fn_for_scaffold(FLAGS.model_dir, FLAGS.checkpoint_path,
                                            FLAGS.model_scope, FLAGS.checkpoint_model_scope,
                                            FLAGS.checkpoint_exclude_scopes, FLAGS.ignore_missing_vars,
                                            name_remap={'/kernel': '/weights', '/bias': '/biases'})

global_anchor_info = dict()

def input_pipeline(dataset_pattern='train-*', is_training=True, batch_size=FLAGS.batch_size):
    out_shape = [FLAGS.train_image_size] * 2
    anchor_creator = anchor_manipulator.AnchorCreator(out_shape,
                                                layers_shapes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
                                                anchor_scales = [(0.1,), (0.2,), (0.375,), (0.55,), (0.725,), (0.9,)],
                                                extra_anchor_scales = [(0.1414,), (0.2739,), (0.4541,), (0.6315,), (0.8078,), (0.9836,)],
                                                anchor_ratios = [(1., 2., .5), (1., 2., 3., .5, 0.3333), (1., 2., 3., .5, 0.3333), (1., 2., 3., .5, 0.3333), (1., 2., .5), (1., 2., .5)],
                                                layer_steps = [8, 16, 32, 64, 100, 300])
    all_anchors, all_num_anchors_depth, all_num_anchors_spatial = anchor_creator.get_all_anchors()

    num_anchors_per_layer = []
    for ind in range(len(all_anchors)):
        num_anchors_per_layer.append(all_num_anchors_depth[ind] * all_num_anchors_spatial[ind])

    prior_scaling = [0.1, 0.1, 0.2, 0.2]
    anchor_encoder_decoder = anchor_manipulator.AnchorEncoder(allowed_borders = [1.0] * 6,
                                                        positive_threshold = FLAGS.match_threshold,
                                                        ignore_threshold = FLAGS.neg_threshold,
                                                        prior_scaling=prior_scaling)

    image_preprocessing_fn = lambda image_, labels_, bboxes_ : ssd_preprocessing.preprocess_image(image_, labels_, bboxes_, out_shape, is_training=is_training, data_format=FLAGS.data_format, output_rgb=False)
    anchor_encoder_fn = lambda glabels_, gbboxes_: anchor_encoder_decoder.encode_all_anchors(glabels_, gbboxes_, all_anchors, all_num_anchors_depth, all_num_anchors_spatial)

    image, _, shape, loc_targets, cls_targets, match_scores = dataset_common.tf1_get_batch(FLAGS.num_classes,
                                                                            batch_size,
                                                                            ('train' if is_training else 'val'),
                                                                            os.path.join(FLAGS.data_dir, dataset_pattern),
                                                                            FLAGS.num_readers,
                                                                            FLAGS.num_preprocessing_threads,
                                                                            image_preprocessing_fn,
                                                                            anchor_encoder_fn,
                                                                            num_epochs=FLAGS.train_epochs,
                                                                            is_training=is_training)
    global global_anchor_info
    global_anchor_info = {'decode_fn': lambda pred : anchor_encoder_decoder.ext_decode_all_anchors(pred, all_anchors, all_num_anchors_depth, all_num_anchors_spatial),
                        'num_anchors_per_layer': num_anchors_per_layer,
                        'all_num_anchors_depth': all_num_anchors_depth }

    return image, {'shape': shape, 'loc_targets': loc_targets, 'cls_targets': cls_targets, 'match_scores': match_scores}


def modified_smooth_l1(bbox_pred, bbox_targets, bbox_inside_weights=1., bbox_outside_weights=1., sigma=1.):
    with tf.name_scope('smooth_l1', [bbox_pred, bbox_targets]):
        sigma2 = sigma * sigma

        inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

        smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
        smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
        smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                                  tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

        outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

        return outside_mul
    
def build_model_graph(features, labels, params, is_training):
    """
    构建完整的计算图，返回 loss, train_op, predictions, metrics 等
    """
    loc_targets = labels['loc_targets']
    cls_targets = labels['cls_targets']

    global global_anchor_info
    decode_fn = global_anchor_info['decode_fn']
    num_anchors_per_layer = global_anchor_info['num_anchors_per_layer']
    all_num_anchors_depth = global_anchor_info['all_num_anchors_depth']

    with tf.variable_scope(params['model_scope'], default_name=None, values=[features], reuse=tf.AUTO_REUSE):
        ssd_backbone = ssd_net.SSDBackbone(
            FLAGS.backbone,
            training=is_training,
            data_format=params['data_format'])
        feature_layers = ssd_backbone.forward(features)
        location_pred, cls_pred = ssd_net.multibox_head(feature_layers, params['num_classes'], all_num_anchors_depth, data_format=params['data_format'])

        if params['data_format'] == 'channels_first':
            cls_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in cls_pred]
            location_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in location_pred]

        cls_pred = [tf.reshape(pred, [tf.shape(features)[0], -1, params['num_classes']]) for pred in cls_pred]
        location_pred = [tf.reshape(pred, [tf.shape(features)[0], -1, 4]) for pred in location_pred]

        cls_pred = tf.concat(cls_pred, axis=1)
        location_pred = tf.concat(location_pred, axis=1)

        cls_pred = tf.reshape(cls_pred, [-1, params['num_classes']])
        location_pred = tf.reshape(location_pred, [-1, 4])

    with tf.device('/cpu:0'):
        with tf.control_dependencies([cls_pred, location_pred]):
            with tf.name_scope('post_forward'):
                bboxes_pred = tf.map_fn(
                    lambda _preds: decode_fn(_preds),
                    tf.reshape(location_pred, [tf.shape(features)[0], -1, 4]),
                    dtype=[tf.float32] * len(num_anchors_per_layer),
                    back_prop=False
                )

                bboxes_pred = [tf.reshape(preds, [-1, 4]) for preds in bboxes_pred]
                bboxes_pred = tf.concat(bboxes_pred, axis=0)

                flaten_cls_targets = tf.reshape(cls_targets, [-1])
                flaten_loc_targets = tf.reshape(loc_targets, [-1, 4])

                positive_mask = flaten_cls_targets > 0
                batch_n_positives = tf.count_nonzero(cls_targets, -1)
                batch_negtive_mask = tf.equal(cls_targets, 0)
                batch_n_negtives = tf.count_nonzero(batch_negtive_mask, -1)

                batch_n_neg_select = tf.cast(params['negative_ratio'] * tf.cast(batch_n_positives, tf.float32), tf.int32)
                batch_n_neg_select = tf.minimum(batch_n_neg_select, tf.cast(batch_n_negtives, tf.int32))

                predictions_for_bg = tf.nn.softmax(tf.reshape(cls_pred, [tf.shape(features)[0], -1, params['num_classes']]))[:, :, 0]
                prob_for_negtives = tf.where(
                    batch_negtive_mask,
                    0. - predictions_for_bg,
                    0. - tf.ones_like(predictions_for_bg)
                )
                topk_prob_for_bg, _ = tf.nn.top_k(prob_for_negtives, k=tf.shape(prob_for_negtives)[1])
                score_at_k = tf.gather_nd(topk_prob_for_bg, tf.stack([tf.range(tf.shape(features)[0]), batch_n_neg_select - 1], axis=-1))
                selected_neg_mask = prob_for_negtives >= tf.expand_dims(score_at_k, axis=-1)

                final_mask = tf.stop_gradient(
                    tf.logical_or(
                        tf.reshape(tf.logical_and(batch_negtive_mask, selected_neg_mask), [-1]),
                        positive_mask
                    )
                )

                cls_pred_masked = tf.boolean_mask(cls_pred, final_mask)
                location_pred_masked = tf.boolean_mask(location_pred, tf.stop_gradient(positive_mask))
                flaten_cls_targets_masked = tf.boolean_mask(
                    tf.clip_by_value(flaten_cls_targets, 0, params['num_classes']), final_mask)
                flaten_loc_targets_masked = tf.stop_gradient(
                    tf.boolean_mask(flaten_loc_targets, positive_mask))

                predictions = {
                    'classes': tf.argmax(cls_pred_masked, axis=-1),
                    'probabilities': tf.reduce_max(tf.nn.softmax(cls_pred_masked, name='softmax_tensor'), axis=-1),
                    # 'loc_predict': bboxes_pred
                }

                cls_accuracy = tf.metrics.accuracy(flaten_cls_targets_masked, predictions['classes'])

    # 损失计算
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        labels=flaten_cls_targets_masked, logits=cls_pred_masked) * (params['negative_ratio'] + 1.)
    tf.identity(cross_entropy, name='cross_entropy_loss')

    loc_loss = modified_smooth_l1(location_pred_masked, flaten_loc_targets_masked, sigma=1.)
    loc_loss = tf.reduce_mean(tf.reduce_sum(loc_loss, axis=-1), name='location_loss')
    tf.losses.add_loss(loc_loss)

    # L2 正则化
    l2_loss_vars = []
    for var in tf.trainable_variables():
        if '_bn' not in var.name:
            if 'conv4_3_scale' not in var.name:
                l2_loss_vars.append(tf.nn.l2_loss(var))
            else:
                l2_loss_vars.append(tf.nn.l2_loss(var) * 0.1)
    l2_loss = tf.multiply(params['weight_decay'], tf.add_n(l2_loss_vars), name='l2_loss')
    total_loss = tf.add(cross_entropy + loc_loss, l2_loss, name='total_loss')

    # 优化器
    global_step = tf.train.get_or_create_global_step()
    lr_values = [params['learning_rate'] * decay for decay in params['lr_decay_factors']]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32),
        [int(_) for _ in params['decay_boundaries']],
        lr_values
    )
    truncated_learning_rate = tf.maximum(learning_rate, params['end_learning_rate'], name='learning_rate')

    optimizer = tf.train.MomentumOptimizer(learning_rate=truncated_learning_rate, momentum=params['momentum'])
    if params['use_amp']:
        optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(total_loss, global_step=global_step)

    return total_loss, train_op, predictions, cls_accuracy, truncated_learning_rate

def validate_batch_size_for_multi_gpu(batch_size):
    """For multi-gpu, batch-size must be a multiple of the number of
    available GPUs.

    Note that this should eventually be handled by replicate_model_fn
    directly. Multi-GPU support is currently experimental, however,
    so doing the work here until that feature is in place.
    """
    if FLAGS.multi_gpu:
        from tensorflow.python.client import device_lib

        local_device_protos = device_lib.list_local_devices()
        num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
        if not num_gpus:
            raise ValueError('Multi-GPU mode was specified, but no GPUs '
                            'were found. To use CPU, run --multi_gpu=False.')

        remainder = batch_size % num_gpus
        if remainder:
            err = ('When running with multiple GPUs, batch size '
                    'must be a multiple of the number of available GPUs. '
                    'Found {} GPUs with a batch size of {}; try --batch_size={} instead.'
                    ).format(num_gpus, batch_size, batch_size - remainder)
            raise ValueError(err)
        return num_gpus
    return 0

def parse_comma_list(args):
    return [float(s.strip()) for s in args.split(',')]

def main(_):
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    try:
        from dltest import show_training_arguments
        show_training_arguments(FLAGS)
    except:
        pass

    # 参数
    num_gpus = validate_batch_size_for_multi_gpu(FLAGS.batch_size)
    params = {
        'num_gpus': num_gpus,
        'data_format': FLAGS.data_format,
        'batch_size': FLAGS.batch_size,
        'model_scope': FLAGS.model_scope,
        'num_classes': FLAGS.num_classes,
        'negative_ratio': FLAGS.negative_ratio,
        'match_threshold': FLAGS.match_threshold,
        'neg_threshold': FLAGS.neg_threshold,
        'weight_decay': FLAGS.weight_decay,
        'momentum': FLAGS.momentum,
        'learning_rate': FLAGS.learning_rate,
        'end_learning_rate': FLAGS.end_learning_rate,
        'decay_boundaries': parse_comma_list(FLAGS.decay_boundaries),
        'lr_decay_factors': parse_comma_list(FLAGS.lr_decay_factors),
        'use_amp': FLAGS.use_amp,
    }

    # 构建图-数据
    features, labels = input_pipeline(
        dataset_pattern='train-*',
        is_training=True,
        batch_size=FLAGS.batch_size
    )

    # 构建图-模型及模型后处理
    total_loss, train_op, predictions, cls_metrics, lr = build_model_graph(features, labels, params, is_training=True)

    # Session 配置
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        intra_op_parallelism_threads=FLAGS.num_cpu_threads,
        inter_op_parallelism_threads=FLAGS.num_cpu_threads,
        gpu_options=gpu_options
    )

    scaffold = tf.train.Scaffold(init_fn=get_init_fn())

    # 训练循环
    with tf.train.MonitoredTrainingSession(
        master='',
        is_chief=True,
        checkpoint_dir=FLAGS.model_dir,
        save_checkpoint_secs=None,
        scaffold=scaffold,
        config=config
    ) as sess:

        step_ = 0
        print('Starting a training cycle.')

        while step_ < FLAGS.max_number_of_steps:
            try:
                _, total_loss_, lr_, step_, acc_ = sess.run([train_op, total_loss, lr, tf.train.get_global_step(), cls_metrics[1]])
                
                if step_ % FLAGS.log_every_n_steps == 0:
                    tf.logging.info('global_step %d: loss = %.4f, lr = %.6f, acc = %.4f', step_, total_loss_, lr_, acc_)

            except tf.errors.OutOfRangeError:
                tf.logging.info('Epoch finished after %d steps.', step_)
                break

    print('Training completed.')

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()