from .common_ops import *


def resnet(is_training, num_classes=None, **kwargs):

    version = kwargs.get('version', 'v1')
    size = kwargs.get('size', 50)
    if version.lower() == 'v1d':
        deepbase = True
        avg_down = True
    else:
        deepbase = False
        avg_down = False
    dropout_ratio = kwargs.get('dropout_ratio', 0.0)
    norm_name = kwargs.get('norm_name', 'batch_norm')
    bn_decay = kwargs.get('bn_decay', 0.99)
    r_max = kwargs.get('r_max', 3)
    d_max = kwargs.get('d_max', 5)
    use_se = kwargs.get('use_se', False)
    se_reduction = kwargs.get('se_reduction', 16)

    stage_blocks = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }[size]
    output_channels = {
        18: [64, 128, 256, 512],
        34: [64, 128, 256, 512],
        50: [256, 512, 1024, 2048],
        101: [256, 512, 1024, 2048],
        152: [256, 512, 1024, 2048],
    }[size]
    num_stages = len(stage_blocks)

    norm_fn = get_normalizer_fn(
        norm_name,
        is_training=is_training,
        bn_decay=0.99,
        r_max=r_max,
        d_max=d_max
    )

    if 'v1' in version:
        net_name = 'ResNetV1'
        if size < 50:
            block_name = 'resblockv1'
        else:
            block_name = 'bottleneckv1'
    elif 'v2' in version:
        net_name = 'ResNetV2'
        if size < 50:
            block_name = 'resblockv2'
        else:
            block_name = 'bottleneckv2'
    else:
        raise NotImplementedError

    def conv_layer(inputs, filters, kernel_size=3, stride=1, padding='SAME', scope=None):
        x = conv2d(inputs, filters, kernel_size=kernel_size, stride=stride,
            use_bias=False, padding=padding, scope=scope)
        x = norm_fn(x, scope=scope)
        x = relu(x)
        return x

    if use_se:
        def squeeze_excite(
            inputs,
            scope=None):
            with tf.variable_scope(scope, default_name='squeeze_excite') as s, \
                 tf.name_scope(s.original_name_scope):
                in_channel = inputs.get_shape().as_list()[3]
                avgpool = global_avg_pooling(inputs, keep_dims=True, name="avgpool")
                squeeze = conv2d(avgpool, in_channel//se_reduction,
                    kernel_size=1, stride=1, use_bias=True, scope='squeeze')
                squeeze = relu(squeeze)
                excite = conv2d(squeeze, in_channel,
                    kernel_size=1, stride=1, use_bias=True, scope='excite')
                excite = sigmoid(excite)
                return inputs * excite
    else:
        squeeze_excite = None
    # A single block for ResNet v1.
    if 'resblockv1' == block_name:
        def block_fn(inputs, out_channel, kernel_size=3, stride=1, scope=None):
            with tf.variable_scope(scope, default_name='ResidualBlockV1') as s, \
                 tf.name_scope(s.original_name_scope):
                in_channel = inputs.get_shape().as_list()[3]
                if in_channel != out_channel:
                    if avg_down and stride != 1:
                        shortcut = avg_pooling(inputs, kernel_size=stride, stride=stride)
                        shortcut = conv2d(shortcut, out_channel,
                            kernel_size=1, stride=1, use_bias=False,
                            scope='projection_shortcut')
                    else:
                        shortcut = conv2d(inputs, out_channel,
                            kernel_size=1, stride=stride, use_bias=False,
                            scope='projection_shortcut')
                    shortcut = norm_fn(shortcut, scope='projection_shortcut')
                else:
                    shortcut = inputs
                x = conv_layer(inputs, out_channel,
                    kernel_size=kernel_size, stride=stride, scope='conv_1')
                x = conv2d(x, out_channel,
                    kernel_size=kernel_size, stride=1, use_bias=False, scope='conv_2')
                x = norm_fn(x, scope='conv_2')
                if squeeze_excite:
                    x = squeeze_excite(x, scope='squeeze_excite')

                return relu(x + shortcut)
    # A single block for ResNet v1, with a bottleneck.
    # Bottleneck places the stride for downsampling at 3x3 convolution(conv2)
    # while original implementation places the stride at the first 1x1 convolution(conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    if 'bottleneckv1' == block_name:
        def block_fn(inputs, out_channel, kernel_size=3, stride=1, reduction=4, scope=None):
            with tf.variable_scope(scope, default_name='BottleneckBlockV1') as s, \
                 tf.name_scope(s.original_name_scope):
                in_channel = inputs.get_shape().as_list()[3]
                if in_channel != out_channel:
                    if avg_down and stride != 1:
                        shortcut = avg_pooling(inputs, kernel_size=stride, stride=stride)
                        shortcut = conv2d(shortcut, out_channel,
                            kernel_size=1, stride=1, use_bias=False,
                            scope='projection_shortcut')
                    else:
                        shortcut = conv2d(inputs, out_channel,
                            kernel_size=1, stride=stride, use_bias=False,
                            scope='projection_shortcut')
                    shortcut = norm_fn(shortcut, scope='projection_shortcut')
                else:
                    shortcut = inputs
                x = conv_layer(inputs, out_channel//reduction,
                    kernel_size=1, stride=1, scope='reduction')
                x = conv_layer(x, out_channel//reduction,
                    kernel_size=kernel_size, stride=stride, scope='bottleneck')
                x = conv2d(x, out_channel,
                    kernel_size=1, stride=1, use_bias=False, scope='expansion')
                x = norm_fn(x, scope='expansion')
                if squeeze_excite:
                    x = squeeze_excite(x, scope='squeeze_excite')
                return relu(x + shortcut)
    # A single block for ResNet v2.
    if 'resblockv2' == block_name:
        def block_fn(inputs, out_channel, kernel_size=3, stride=1, scope=None):
            with tf.variable_scope(scope, default_name='ResidualBlockV2') as s, \
                 tf.name_scope(s.original_name_scope):
                in_channel = inputs.get_shape().as_list()[3]
                if in_channel != out_channel:
                    if avg_down and stride != 1:
                        shortcut = avg_pooling(inputs, kernel_size=stride, stride=stride)
                        shortcut = conv2d(shortcut, out_channel,
                            kernel_size=1, stride=1, use_bias=False,
                            scope='projection_shortcut')
                    else:
                        shortcut = conv2d(inputs, out_channel,
                            kernel_size=1, stride=stride, use_bias=False,
                            scope='projection_shortcut')
                else:
                    shortcut = inputs
                x = norm_fn(inputs, scope='norm_inputs')
                x = relu(x)
                x = conv_layer(x, out_channel,
                    kernel_size=kernel_size, stride=stride, scope='conv_1')
                x = conv2d(x, out_channel,
                    kernel_size=kernel_size, stride=1, use_bias=False, scope='conv_2')
                if squeeze_excite:
                    x = squeeze_excite(x, scope='squeeze_excite')

                return x + shortcut
    # A single block for ResNet v2, with a bottleneck.
    if 'bottleneckv2' == block_name:
        def block_fn(inputs, out_channel, kernel_size=3, stride=1, reduction=4, scope=None):
            with tf.variable_scope(scope, default_name='BottleneckBlockV2') as s, \
                 tf.name_scope(s.original_name_scope):
                in_channel = inputs.get_shape().as_list()[3]
                if in_channel != out_channel:
                    if avg_down and stride != 1:
                        shortcut = avg_pooling(inputs, kernel_size=stride, stride=stride)
                        shortcut = conv2d(shortcut, out_channel,
                            kernel_size=1, stride=1, use_bias=False,
                            scope='projection_shortcut')
                    else:
                        shortcut = conv2d(inputs, out_channel,
                            kernel_size=1, stride=stride, use_bias=False,
                            scope='projection_shortcut')
                else:
                    shortcut = inputs
                x = norm_fn(inputs, scope='norm_inputs')
                x = relu(x)
                x = conv_layer(x, out_channel//reduction,
                    kernel_size=1, stride=1, scope='reduction')
                x = conv_layer(x, out_channel//reduction,
                    kernel_size=kernel_size, stride=stride, scope='bottleneck')
                x = conv2d(x, out_channel,
                    kernel_size=1, stride=1, use_bias=False, scope='expansion')
                if squeeze_excite:
                    x = squeeze_excite(x, scope='squeeze_excite')
                return x + shortcut

    def first_layer(inputs, filters=64, scope='first_layer'):
        with tf.variable_scope(scope):
            if deepbase:
                x = conv_layer(inputs, filters//2, kernel_size=3, stride=2, scope='Conv_1')
                x = conv_layer(x, filters//2, kernel_size=3, stride=1, scope='Conv_2')
                if net_name == 'ResNetV2':
                    x = conv2d(x, filters,
                        kernel_size=3, stride=1, use_bias=False, scope='Conv_3')
                else:
                    x = conv_layer(x, filters, kernel_size=3, stride=1, scope='Conv_3')
            else:
                if net_name == 'ResNetV2':
                    x = conv2d(inputs, filters,
                        kernel_size=7, stride=2, use_bias=False, scope='Conv_1')
                else:
                    x = conv_layer(inputs, filters, kernel_size=7, stride=2, scope='Conv_1')
            return x

    end_points = []

    def forward(inputs):

        with tf.variable_scope(net_name, reuse=tf.AUTO_REUSE):
            net = first_layer(inputs, scope='first_layer')
            #end_points['down_1'] = net
            end_points.append(net)
            net = max_pooling(net, kernel_size=3, stride=2)
            # stage 1,2,3,4
            for stage_idx in range(num_stages):
                stage_scope = 'stage_{}'.format(stage_idx+1)
                num_blocks = stage_blocks[stage_idx]
                out_channel = output_channels[stage_idx]
                with tf.variable_scope(stage_scope):
                    for block_idx in range(num_blocks):
                        block_scope = 'block_{}'.format(block_idx+1)
                        stride = 2 if (stage_idx > 0 and block_idx == 0) else 1
                        net = block_fn(net, out_channel,
                            kernel_size=3, stride=stride, scope=block_scope)
                #end_points['down_%d' %(stage_idx+2)] = net
                end_points.append(net)

            if net_name == 'ResNetV2':
                net = norm_fn(net, scope='postnorm')
                net = relu(net)
                #end_points['down_%d' %((num_stages-1)+2)] = net
                end_points[-1] = net

            if not num_classes:
                return end_points

            net = global_avg_pooling(net)

            if dropout_ratio:
                net = dropout(net, is_training=is_training, dropout_ratio=dropout_ratio)

            logits = conv2d(net, num_classes, kernel_size=1, use_bias=True, scope='classification')
            logits = tf.squeeze(logits, [1, 2], name='logits')

            return logits

    return forward


def resnet18(is_training, num_classes=None, **kwargs):
    return resnet(is_training, num_classes, size=18, **kwargs)


def resnet34(is_training, num_classes=None, **kwargs):
    return resnet(is_training, num_classes, size=34, **kwargs)


def resnet50(is_training, num_classes=None, **kwargs):
    return resnet(is_training, num_classes, size=50, **kwargs)


def resnet101(is_training, num_classes=None, **kwargs):
    return resnet(is_training, num_classes, size=101, **kwargs)
