import math
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


def fixed_padding(inputs, kernel_size, rate=1):
    kernel_size_effective = [kernel_size[0] + (kernel_size[0] - 1) * (rate - 1),
                            kernel_size[1] + (kernel_size[1] - 1) * (rate - 1)]

    pad_total = [kernel_size_effective[0] - 1, kernel_size_effective[1] - 1]
    pad_beg = [pad_total[0] // 2, pad_total[1] // 2]
    pad_end = [pad_total[0] - pad_beg[0], pad_total[1] - pad_beg[1]]

    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg[0], pad_end[0]],
                                    [pad_beg[1], pad_end[1]], [0, 0]])
    return padded_inputs


def conv2d(
    inputs,
    filters,
    kernel_size=3,
    stride=1,
    padding='SAME',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_in', distribution='truncated_normal'),
    bias_initializer=tf.constant_initializer(0.0),
    dilation_rate=1,
    scope=None):
    with tf.variable_scope(scope, default_name='Conv2D') as s, \
         tf.name_scope(s.original_name_scope):
        in_channel = inputs.get_shape().as_list()[3]
        if type(kernel_size).__name__ == 'int':
            kernel_shape = [kernel_size, kernel_size, in_channel, filters]
        else:
            kernel_shape = kernel_size + [in_channel, filters]

        kernel = tf.get_variable(
            'kernel', shape=kernel_shape, dtype=tf.float32,
            initializer=kernel_initializer, trainable=True)

        if padding.lower() == 'same':
            inputs = fixed_padding(inputs, kernel_size=kernel_shape[0:2], rate=dilation_rate)

        if dilation_rate > 1:
            outputs = tf.nn.atrous_conv2d(inputs, kernel, rate=dilation_rate, padding='VALID')
        else:
            strides = [1, stride, stride, 1]
            outputs = tf.nn.conv2d(
                inputs, kernel, strides=strides, padding='VALID',
                use_cudnn_on_gpu=True, name='convolution')
        if use_bias:
            b = tf.get_variable(
                'bias', shape=[filters], dtype=tf.float32,
                initializer=bias_initializer, trainable=True)
            outputs = tf.nn.bias_add(outputs, b)
        return outputs


def depthwise_conv2d(
    inputs,
    filters=None,
    kernel_size=3,
    stride=1,
    padding='SAME',
    depth_multiplier=1,
    use_bias=False,
    kernel_initializer=tf.keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_in', distribution='truncated_normal'),
    bias_initializer=tf.constant_initializer(0.0),
    dilation_rate=1,
    scope=None):
    with tf.variable_scope(scope, default_name='DepthwiseConv2D') as s, \
         tf.name_scope(s.original_name_scope):
        in_channel = inputs.get_shape().as_list()[3]
        if filters:
            assert filters % in_channel == 0
            depth_multiplier = filters // in_channel
        if type(kernel_size).__name__ == 'int':
            kernel_shape = [kernel_size, kernel_size, in_channel, depth_multiplier]
        else:
            kernel_shape = kernel_size + [in_channel, depth_multiplier]

        kernel = tf.get_variable(
            'depthwise_kernel', shape=kernel_shape, dtype=tf.float32,
            initializer=kernel_initializer, trainable=True)

        if padding.lower() == 'same':
            inputs = fixed_padding(inputs, kernel_size=kernel_shape[0:2], rate=dilation_rate)

        if dilation_rate > 1:
            strides = [1, 1, 1, 1]
            outputs = tf.nn.depthwise_conv2d(
                inputs, kernel, strides=strides, padding='VALID', rate=dilation_rate)
        else:
            strides = [1, stride, stride, 1]
            # param filter of tf.nn.depthwise_conv2d: [filter_height, filter_width, in_channels, channel_multiplier]
            outputs = tf.nn.depthwise_conv2d(
                inputs, kernel, strides=strides, padding='VALID', rate=None)
        if use_bias:
            b = tf.get_variable(
                'depthwise_bias', shape=[in_channel], dtype=tf.float32,
                initializer=bias_initializer, trainable=True)
            outputs = tf.nn.bias_add(outputs, b)
        return outputs


# Flatten the tensor except the first dimension.
def _batch_flatten(x):
    shape = x.get_shape().as_list()[1:]
    if None not in shape:
        return tf.reshape(x, [-1, int(np.prod(shape))])
    return tf.reshape(x, tf.stack([tf.shape(x)[0], -1]))


def fully_connected(
    inputs,
    filters,
    use_bias=True,
    kernel_initializer=tf.keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_in', distribution='truncated_normal'),
    bias_initializer=tf.constant_initializer(0.0),
    scope=None):
    with tf.variable_scope(scope, default_name='Conv2D') as s, \
         tf.name_scope(s.original_name_scope):
        inputs = _batch_flatten(inputs)
        in_channel = inputs.get_shape().as_list()[1]
        kernel_shape = [in_channel, filters]

        kernel = tf.get_variable('kernel', shape=kernel_shape, dtype=tf.float32, initializer=kernel_initializer, trainable=True)
        outputs = tf.matmul(inputs, kernel)
        if use_bias:
            b = tf.get_variable('bias', shape=[filters], dtype=tf.float32, initializer=bias_initializer, trainable=True)
            outputs = tf.nn.bias_add(outputs, b)
        return outputs


def get_normalizer_fn(
    norm_name,
    is_training,
    bn_decay=0.99,
    ema_update=True,
    r_max=3,
    d_max=5,
    group=8):

    def normalizer_fn(inputs, scope=''):
        if 'batch_norm' == norm_name.lower():
            if type(is_training) is tf.Tensor:
                return tf.cond(
                    is_training,
                    lambda: batch_normalization(inputs=inputs, is_training=True,
                        bn_decay=bn_decay, ema_update=ema_update, scope=scope, reuse=None),
                    lambda: batch_normalization(inputs=inputs, is_training=False,
                        bn_decay=bn_decay, ema_update=ema_update, scope=scope, reuse=True)
                )
            else:
                return batch_normalization(inputs=inputs, is_training=is_training,
                    bn_decay=bn_decay, ema_update=ema_update, scope=scope, reuse=None)
        elif 'batch_renorm' == norm_name.lower():
            if type(is_training) is tf.Tensor:
                return tf.cond(
                    is_training,
                    lambda: batch_renormalization(inputs=inputs, is_training=True,
                        r_max=r_max, d_max=d_max, bn_decay=bn_decay,
                        ema_update=ema_update, scope=scope, reuse=None),
                    lambda: batch_renormalization(inputs=inputs, is_training=False,
                        r_max=r_max, d_max=d_max, bn_decay=bn_decay,
                        ema_update=ema_update, scope=scope, reuse=True)
                )
            else:
                return batch_renormalization(inputs=inputs, is_training=is_training,
                    r_max=r_max, d_max=d_max, bn_decay=bn_decay,
                    ema_update=ema_update, scope=scope, reuse=None)
        elif 'group_norm' == norm_name.lower():
            if type(is_training) is tf.Tensor:
                return group_norm(inputs, is_training=True, group=group, scope=scope)
            else:
                return group_norm(inputs, is_training=is_training, group=group, scope=scope)
        elif 'instance_norm' == norm_name.lower():
            if type(is_training) is tf.Tensor:
                return instance_norm(inputs, is_training=True, scope=scope)
            else:
                return instance_norm(inputs, is_training=is_training, scope=scope)
        else:
            return tf.identity(inputs)

    return normalizer_fn


def moments(x, axes, keep_dims=False, is_training=False, name=None):
    with ops.name_scope(name, "moments", [x, axes]):
        mean = math_ops.reduce_mean(x, axes, keepdims=True, name="mean")
        if is_training:
            squared_difference = math_ops.squared_difference(x, array_ops.stop_gradient(mean))
        else:
            squared_difference = math_ops.squared_difference(x, mean)
        variance = math_ops.reduce_mean(squared_difference, axes, keepdims=True, name="variance")
        if not keep_dims:
            mean = array_ops.squeeze(mean, axes)
            variance = array_ops.squeeze(variance, axes)
        return (mean, variance)


def batch_normalization(
    inputs,
    is_training,
    epsilon=1e-5,
    bn_decay=0.99,
    ema_update=True,
    scope=None,
    reuse=None):

    if scope:
        scope = scope + '/BatchNorm'
    with tf.variable_scope(scope, default_name='BatchNorm', reuse=reuse) as s, \
         tf.name_scope(s.original_name_scope):
        C = inputs.get_shape().as_list()[3]
        # gamma: a trainable scale factor
        gamma = tf.get_variable("gamma", [C],
            initializer=tf.constant_initializer(1.0), trainable=True)
        # beta: a trainable shift value
        beta = tf.get_variable("beta", [C],
            initializer=tf.constant_initializer(0.0), trainable=True)
        moving_mean = tf.get_variable("moving_mean", [C],
            initializer=tf.constant_initializer(0.0), trainable=False)
        moving_variance = tf.get_variable("moving_variance", [C],
            initializer=tf.constant_initializer(1.0), trainable=False)
        # use batch statistics
        if is_training:
            mean, var = moments(inputs, [0,1,2], keep_dims=True, is_training=is_training)
            mean = tf.reshape(mean, [C])
            var = tf.reshape(var, [C])
            # update moving_mean and moving_variance
            if ema_update:
                update_moving_mean = tf.assign(
                    moving_mean, moving_mean * bn_decay + mean * (1 - bn_decay))
                update_moving_variance = tf.assign(
                    moving_variance, moving_variance * bn_decay + var * (1 - bn_decay))
                control_inputs = [update_moving_mean, update_moving_variance]
            else:
                control_inputs = []
            with tf.control_dependencies(control_inputs):
                output = tf.nn.batch_normalization(
                    inputs, mean, var, offset=beta, scale=gamma, variance_epsilon=epsilon)
        # use EMA statistics
        else:
            output = tf.nn.batch_normalization(
                inputs, moving_mean, moving_variance, offset=beta, scale=gamma,
                variance_epsilon=epsilon)

    return output


def batch_renormalization(
    inputs,
    is_training,
    r_max=3,
    d_max=5,
    epsilon=1e-5,
    bn_decay=0.99,
    ema_update=False,
    scope=None,
    reuse=None):

    if scope:
        scope = scope + '/BatchNorm'
    with tf.variable_scope(scope, default_name='BatchNorm', reuse=reuse) as s, \
         tf.name_scope(s.original_name_scope):
        C = inputs.get_shape().as_list()[3]
        # gamma: a trainable scale factor
        gamma = tf.get_variable("gamma", [C],
            initializer=tf.constant_initializer(1.0), trainable=True)
        # beta: a trainable shift value
        beta = tf.get_variable("beta", [C],
            initializer=tf.constant_initializer(0.0), trainable=True)
        moving_mean = tf.get_variable("moving_mean", [C],
            initializer=tf.constant_initializer(0.0), trainable=False)
        moving_variance = tf.get_variable("moving_variance", [C],
            initializer=tf.constant_initializer(1.0), trainable=False)
        # use batch statistics
        if is_training:
            mean, var = moments(inputs, [0,1,2], keep_dims=True, is_training=is_training)
            mean = tf.reshape(mean, [C])
            var = tf.reshape(var, [C])
            std = math_ops.sqrt(var + epsilon)

            r = std / (math_ops.sqrt(moving_variance + epsilon))
            r = array_ops.stop_gradient(tf.clip_by_value(r, 1/r_max, r_max))

            d = (mean - moving_mean) / math_ops.sqrt(moving_variance + epsilon)
            d = array_ops.stop_gradient(tf.clip_by_value(d, -d_max, d_max))
            # update moving_mean and moving_variance
            if ema_update:
                update_moving_mean = tf.assign(moving_mean, moving_mean * bn_decay + mean * (1 - bn_decay))
                update_moving_variance = tf.assign(moving_variance, moving_variance * bn_decay + var * (1 - bn_decay))
                control_inputs = [update_moving_mean, update_moving_variance]
            else:
                control_inputs = []

            batch_normed_output = (inputs - mean) / std
            with tf.control_dependencies(control_inputs):
                output = (batch_normed_output * r + d) * gamma + beta
        # use EMA statistics
        else:
            output = tf.nn.batch_normalization(
                inputs, moving_mean, moving_variance, offset=beta, scale=gamma,
                variance_epsilon=epsilon)

        return output


def group_norm(inputs, group=8, epsilon=1e-5, is_training=False, scope=None):
    if scope:
        scope = scope + '/GroupNorm'
    with tf.variable_scope(scope, default_name='GroupNorm') as s, \
         tf.name_scope(s.original_name_scope):
        C = inputs.get_shape().as_list()[3]
        orig_shape = tf.shape(inputs)
        H, W = orig_shape[1], orig_shape[2]
        G = min(group, C)

        x = tf.reshape(inputs, [-1, H, W, G, C//G])
        mean, var = moments(x, [1, 2, 4], keep_dims=True, is_training=is_training)

        gamma = tf.get_variable('gamma', shape=[C], dtype=tf.float32,
            initializer=tf.constant_initializer(1.0), trainable=True)
        beta = tf.get_variable('beta', shape=[C], dtype=tf.float32,
            initializer=tf.constant_initializer(0.0), trainable=True)
        gamma = tf.reshape(gamma, [1, 1, 1, G, C//G])
        beta = tf.reshape(beta, [1, 1, 1, G, C//G])

        output = tf.nn.batch_normalization(
            x, mean, var, offset=beta, scale=gamma, variance_epsilon=epsilon)
        output = tf.reshape(output, orig_shape)
        return output


def instance_norm(inputs, epsilon=1e-5, is_training=False, scope=None):
    if scope:
        scope = scope + '/InstanceNorm'
    with tf.variable_scope(scope, default_name='InstanceNorm') as s, \
         tf.name_scope(s.original_name_scope):
        B = tf.shape(inputs)[0]
        C = inputs.get_shape().as_list()[-1]

        gamma = tf.get_variable('gamma', shape=[C], dtype=tf.float32,
            initializer=tf.constant_initializer(1.0), trainable=True)
        beta = tf.get_variable('beta', shape=[C], dtype=tf.float32,
            initializer=tf.constant_initializer(0.0), trainable=True)
        gamma = tf.reshape(gamma, [1, 1, 1, C])
        beta = tf.reshape(beta, [1, 1, 1, C])

        mean, var = moments(inputs, [1, 2], keep_dims=True, is_training=is_training)
        output = tf.nn.batch_normalization(
            inputs, mean, var, offset=beta, scale=gamma, variance_epsilon=epsilon)
        return output


def relu(x, name="relu"):
    return tf.nn.relu(x, name=name)


def relu6(x, name="relu6"):
    return tf.nn.relu6(x, name=name)


# x = tf.where(x < 0.0, leak * x, x)
def leaky_relu(x, leak=0.01, name="leaky_relu"):
    return tf.nn.leaky_relu(x, alpha=leak, name=name)


# x = tf.where(x > 0.0, x, alpha * tf.exp(x) - alpha)
# alpha =1.0 by default
def elu(x, name='elu'):
    return tf.nn.elu(x, name=name)


# alpha = 1.6732632423543772848170429916717
# scale = 1.0507009873554804934193349852946
# x = scale * tf.where(x > 0.0, x, alpha * tf.exp(x) - alpha)
def selu(x, name='selu'):
    return tf.nn.selu(x)


def hard_swish(x):
    return x * tf.nn.relu6(x + 3.0) / 6.0


# x = tf.clip_by_value(x + 3.0, 0.0, 6.0) / 6.0
def hard_sigmoid(x):
    return tf.nn.relu6(x + 3.0) / 6.0


def sigmoid(x):
    return tf.nn.sigmoid(x)


def max_pooling(inputs, kernel_size=2, stride=2, padding='SAME', name='MaxPooling'):
    if type(kernel_size).__name__ == 'int':
        kernel_size = [1, kernel_size, kernel_size, 1]
    else:
        kernel_size = [1] + kernel_size + [1]
    if type(stride).__name__ == 'int':
        strides = [1, stride, stride, 1]
    else:
        strides = [1] + stride + [1]
    return tf.nn.max_pool(inputs, ksize=kernel_size, strides=strides, padding=padding, data_format='NHWC', name=name)


def avg_pooling(inputs, kernel_size=2, stride=2, padding='SAME', name='AveragePooling'):
    if type(kernel_size).__name__ == 'int':
        kernel_size = [1, kernel_size, kernel_size, 1]
    else:
        kernel_size = [1] + kernel_size + [1]
    if type(stride).__name__ == 'int':
        strides = [1, stride, stride, 1]
    else:
        strides = [1] + stride + [1]
    return tf.nn.avg_pool(inputs, ksize=kernel_size, strides=strides, padding=padding, data_format='NHWC', name=name)


def global_avg_pooling(inputs, keep_dims=True, name='GlobalAveragePooling'):
    shape = inputs.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        output = math_ops.reduce_mean(inputs, [1,2], keepdims=True, name=name)
    else:
        kernel_size = [1, shape[1], shape[2], 1]
        output = tf.nn.avg_pool(inputs, ksize=kernel_size, strides=[1, 1, 1, 1], padding='VALID', name=name)
    return output


def get_tensor_size(inputs):
    shape = inputs.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        return tf.shape(inputs)[1:3]
    else:
        return shape[1:3]


def dropout(inputs, is_training, dropout_ratio=0.0, name='Dropout'):
    if type(is_training) is tf.Tensor:
        return tf.cond(
            is_training,
            lambda: tf.nn.dropout(inputs, dropout_ratio, name=name),
            lambda: tf.identity(inputs, name=name)
        )
    elif is_training:
        return tf.nn.dropout(inputs, dropout_ratio, name=name)
    else:
        return tf.identity(inputs, name=name)
