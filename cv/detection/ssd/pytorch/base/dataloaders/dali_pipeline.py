import torch
import os

# DALI import
from .dali_iterator import COCOPipeline
from box_coder import dboxes300_coco

anchors_ltrb_list = dboxes300_coco()("ltrb").numpy().flatten().tolist()

def prebuild_dali_pipeline(config):
    """
        Equivalent to SSD Transformer.
    :param config: configuration
    :return: Dali Pipeline
    """
    train_annotate = os.path.join(config.data_dir, "annotations/bbox_only_instances_train2017.json")
    train_coco_root = os.path.join(config.data_dir, "train2017")
    pipe = COCOPipeline(config.train_batch_size,
                        config.local_rank, train_coco_root,
                        train_annotate, config.n_gpu,
                        anchors_ltrb_list,
                        num_threads=config.num_workers,
                        output_fp16=config.fp16, output_nhwc=config.nhwc,
                        pad_output=config.pad_input, seed=config.local_seed - 2**31,
                        use_nvjpeg=config.use_nvjpeg,
                        dali_cache=config.dali_cache,
                        dali_async=(not config.dali_sync))
    pipe.build()
    return pipe

def build_dali_pipeline(config, training=True, pipe=None):
    # pipe is prebuilt without touching the data
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    train_loader = DALIGenericIterator(pipelines=[pipe],
                                       output_map= ['image', 'bbox', 'label'],
                                       size=pipe.epoch_size()['train_reader'] // config.n_gpu,
                                       auto_reset=True)
    return train_loader, pipe.epoch_size()['train_reader']
