# optimizer
optimizer = dict(type='AdamW', lr=1e-3,betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='poly', power=0.9)
# running settings
runner = dict(type='EpochBasedRunner', max_epochs=1200)
checkpoint_config = dict(interval=50)