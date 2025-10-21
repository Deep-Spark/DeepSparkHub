# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

import logging
import os
import sys
import time
import torch
import torchvision

sys.path.insert(0, '..')
from fp16_utils.fp16util import network_to_half
from utils.utils import AverageMeter
from utils.vis import save_debug_images
from utils.vis import save_valid_image
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size
from dataset import make_test_dataloader
from core.group import HeatmapParser
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    print(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    print('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    print(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )
    values = list(values)
    avg_AP_AR = (values[0] + values[5]) / 2.0
    return avg_AP_AR 


def do_valid(cfg, model, epoch, final_output_dir, fp16=False):
    # switch to eval mode
    model.eval()

    if fp16:
        model = network_to_half(model)
    
    data_loader, test_dataset = make_test_dataloader(cfg)
    
    if cfg.MODEL.NAME == 'pose_hourglass':
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )
    else:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )
    parser = HeatmapParser(cfg)
    all_preds = []
    all_scores = []

    pbar = tqdm(total=len(test_dataset)) if cfg.TEST.LOG_PROGRESS else None
    for i, (images, annos) in enumerate(data_loader):
        assert 1 == images.size(0), 'Test batch size should be 1'

        image = images[0].cpu().numpy()
        # size at scale 1.0
        base_size, center, scale = get_multi_scale_size(
            image, cfg.DATASET.INPUT_SIZE, 1.0, min(cfg.TEST.SCALE_FACTOR)
        )

        with torch.no_grad():
            final_heatmaps = None
            tags_list = []
            for idx, s in enumerate(sorted(cfg.TEST.SCALE_FACTOR, reverse=True)):
                input_size = cfg.DATASET.INPUT_SIZE
                image_resized, center, scale = resize_align_multi_scale(
                    image, input_size, s, min(cfg.TEST.SCALE_FACTOR)
                )
                image_resized = transforms(image_resized)
                image_resized = image_resized.unsqueeze(0).cuda()

                outputs, heatmaps, tags = get_multi_stage_outputs(
                    cfg, model, image_resized, cfg.TEST.FLIP_TEST,
                    cfg.TEST.PROJECT2IMAGE, base_size
                )

                final_heatmaps, tags_list = aggregate_results(
                    cfg, s, final_heatmaps, tags_list, heatmaps, tags
                )

            final_heatmaps = final_heatmaps / float(len(cfg.TEST.SCALE_FACTOR))
            tags = torch.cat(tags_list, dim=4)
            grouped, scores = parser.parse(
                final_heatmaps, tags, cfg.TEST.ADJUST, cfg.TEST.REFINE
            )

            final_results = get_final_preds(
                grouped, center, scale,
                [final_heatmaps.size(3), final_heatmaps.size(2)]
            )

        if cfg.TEST.LOG_PROGRESS:
            pbar.update()

        if i % cfg.PRINT_FREQ == 0:
            prefix = '{}_{}'.format(os.path.join(final_output_dir, 'result_valid'), i)
            # logger.info('=> write {}'.format(prefix))
            save_valid_image(image, final_results, '{}.jpg'.format(prefix), dataset=test_dataset.name)
            # save_debug_images(cfg, image_resized, None, None, outputs, prefix)

        all_preds.append(final_results)
        all_scores.append(scores)
        
    if cfg.TEST.LOG_PROGRESS:
        pbar.close()

    name_values, _ = test_dataset.evaluate(
        cfg, all_preds, all_scores, final_output_dir
    )
    
    avg_AP_AR = _print_name_value(name_values, cfg.MODEL.NAME)
    return avg_AP_AR

