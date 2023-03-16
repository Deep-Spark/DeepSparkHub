model = dict(
    type='DBNet',
    backbone=dict(
        type='dbnet_det.MobileNetV3',
        arch='large',
        # num_stages=3,
        out_indices=(3, 6, 12, 16),
        norm_cfg=dict(type='BN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/pretrain/third_party/mobilenet_v3_large-bc2c3fd3.pth')
        ),
    # backbone=dict(
    #     type='dbnet_det.ResNet',
    #     depth=18,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     frozen_stages=-1,
    #     norm_cfg=dict(type='BN', requires_grad=True),
    #     # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
    #     norm_eval=False,
    #     style='caffe'),
    neck=dict(
        type='FPNC', in_channels=[24, 40, 112, 960], lateral_channels=256),
    bbox_head=dict(
        type='DBHead',
        in_channels=256,
        loss=dict(type='DBLoss', alpha=5.0, beta=10.0, bbce_loss=False),
        postprocessor=dict(type='DBPostprocessor', text_repr_type='quad')),
    train_cfg=None,
    test_cfg=None)



