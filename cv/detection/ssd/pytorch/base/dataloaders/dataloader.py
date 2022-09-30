import torch

from .build_pipeline import prebuild_pipeline, build_pipeline

def create_train_dataloader(config):
    train_pipe = prebuild_pipeline(config)
    train_loader, epoch_size, train_sampler = build_pipeline(config, training=True, pipe=train_pipe)
    return train_loader, epoch_size, train_sampler


def create_eval_dataloader(config):
    val_loader, inv_map, cocoGt = build_pipeline(config, training=False)
    return val_loader, inv_map, cocoGt
