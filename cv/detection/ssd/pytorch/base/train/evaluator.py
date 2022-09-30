from concurrent.futures import ProcessPoolExecutor
import sys

import numpy as np
from pycocotools.cocoeval import COCOeval
import torch
import torch.distributed as dist


import utils
from dataloaders.dataloader import create_eval_dataloader
from dataloaders.prefetcher import eval_prefetcher

import config
from box_coder import build_ssd300_coder

class Evaluator:

    def __init__(self, config):
        self.config = config
        self.eval_count = 0

        self._dataloader = None
        self.fetch_dataloader()

        self.ret = []

        self.overlap_threshold = 0.50
        self.nms_max_detections = 200
        self.encoder = build_ssd300_coder(config.fast_nms)

    def fetch_dataloader(self):
        if self._dataloader is None:
            self._dataloader, self.inv_map, self.cocoGt = create_eval_dataloader(config)
        return self._dataloader

    def evaluate_coco(self, final_results, cocoGt):
        if self.config.use_coco_ext:
            cocoDt = cocoGt.loadRes(final_results, use_ext=True)
            E = COCOeval(cocoGt, cocoDt, iouType='bbox', use_ext=True)
        else:
            cocoDt = cocoGt.loadRes(final_results)
            E = COCOeval(cocoGt, cocoDt, iouType='bbox')
        E.evaluate()
        E.accumulate()
        E.summarize()
        print("Current AP: {:.5f} AP".format(E.stats[0]))
        return E.stats[0]

    def evaluate(self, trainer):
        self.eval_count += 1
        eval_dataloader = eval_prefetcher(iter(self._dataloader),
                                     torch.cuda.current_device(),
                                     config.pad_input,
                                     config.nhwc,
                                     config.fp16)
        trainer.model.eval()
        ret = []
        with torch.no_grad():
            for batch in eval_dataloader:
                img, img_id, img_size = batch
                _, ploc, plabel = trainer.inference(img)

                # torch.save({
                #     "bbox": ploc,
                #     "scores": plabel,
                #     "criteria": self.overlap_threshold,
                #     "max_output": self.nms_max_detections,
                # }, "decode_inputs_{}.pth".format(config.local_rank))
                # exit()

                for idx in range(ploc.shape[0]):
                    # ease-of-use for specific predictions
                    ploc_i = ploc[idx, :, :].unsqueeze(0)
                    plabel_i = plabel[idx, :, :].unsqueeze(0)

                    result = self.encoder.decode_batch(ploc_i, plabel_i, self.overlap_threshold, self.nms_max_detections)[0]

                    htot, wtot = img_size[0][idx].item(), img_size[1][idx].item()
                    loc, label, prob = [r.cpu().numpy() for r in result]
                    for loc_, label_, prob_ in zip(loc, label, prob):
                        ret.append([img_id[idx], loc_[0] * wtot, \
                                    loc_[1] * htot,
                                    (loc_[2] - loc_[0]) * wtot,
                                    (loc_[3] - loc_[1]) * htot,
                                    prob_,
                                    self.inv_map[label_]])

        trainer.model.train()
        ret = np.array(ret).astype(np.float32)
        if self.config.distributed:
            ret_copy = torch.tensor(ret).cuda()
            ret_sizes = [torch.tensor(0).cuda() for _ in range(config.n_gpu)]
            torch.distributed.all_gather(ret_sizes, torch.tensor(ret_copy.shape[0]).cuda())
            max_size = 0
            sizes = []
            for s in ret_sizes:
                max_size = max(max_size, s.item())
                sizes.append(s.item())
            ret_pad = torch.cat([ret_copy, torch.zeros(max_size - ret_copy.shape[0], 7, dtype=torch.float32).cuda()])
            other_ret = [torch.zeros(max_size, 7, dtype=torch.float32).cuda() for i in range(config.n_gpu)]
            torch.distributed.all_gather(other_ret, ret_pad)
            cat_tensors = []
            for i in range(config.n_gpu):
                cat_tensors.append(other_ret[i][:sizes[i]][:])

            final_results = torch.cat(cat_tensors).cpu().numpy()
        else:
            final_results = ret

        if utils.is_main_process():
            eval_ap = self.evaluate_coco(final_results, self.cocoGt)
            return eval_ap
        else:
            return 0

