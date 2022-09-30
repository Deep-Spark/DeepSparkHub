import random
import torch

import torch.nn.functional as F
import numpy as np
import itertools

from math import sqrt


def calc_iou_tensor(box1, box2):
    """ Calculation of IoU based on two boxes tensor,
        Reference to https://github.com/kuangliu/pytorch-ssd
        input:
            box1 (N, 4)
            box2 (M, 4)
        output:
            IoU (N, M)
    """
    N = box1.size(0)
    M = box2.size(0)

    be1 = box1.unsqueeze(1).expand(-1, M, -1)
    be2 = box2.unsqueeze(0).expand(N, -1, -1)

    # Left Top & Right Bottom
    lt = torch.max(be1[:,:,:2], be2[:,:,:2])
    #mask1 = (be1[:,:, 0] < be2[:,:, 0]) ^ (be1[:,:, 1] < be2[:,:, 1])
    #mask1 = ~mask1
    rb = torch.min(be1[:,:,2:], be2[:,:,2:])
    #mask2 = (be1[:,:, 2] < be2[:,:, 2]) ^ (be1[:,:, 3] < be2[:,:, 3])
    #mask2 = ~mask2

    delta = rb - lt
    delta[delta < 0] = 0
    intersect = delta[:,:,0]*delta[:,:,1]
    #*mask1.float()*mask2.float()

    delta1 = be1[:,:,2:] - be1[:,:,:2]
    area1 = delta1[:,:,0]*delta1[:,:,1]
    delta2 = be2[:,:,2:] - be2[:,:,:2]
    area2 = delta2[:,:,0]*delta2[:,:,1]

    iou = intersect/(area1 + area2 - intersect)
    return iou


class DefaultBoxes(object):
    def __init__(self, fig_size, feat_size, steps, scales, aspect_ratios, \
                       scale_xy=0.1, scale_wh=0.2):

        self.feat_size = feat_size
        self.fig_size = fig_size

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh

        # According to https://github.com/weiliu89/caffe
        # Calculation method slightly different from paper
        self.steps = steps
        self.scales = scales

        fk = fig_size/np.array(steps)
        self.aspect_ratios = aspect_ratios

        self.default_boxes = []
        # size of feature and number of feature
        for idx, sfeat in enumerate(self.feat_size):

            sk1 = scales[idx]/fig_size
            sk2 = scales[idx+1]/fig_size
            sk3 = sqrt(sk1*sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            for alpha in aspect_ratios[idx]:
                w, h = sk1*sqrt(alpha), sk1/sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):
                    cx, cy = (j+0.5)/fk[idx], (i+0.5)/fk[idx]
                    self.default_boxes.append((cx, cy, w, h))

        self.dboxes = torch.tensor(self.default_boxes, dtype=torch.float)
        self.dboxes.clamp_(min=0, max=1)
        # For IoU calculation
        self.dboxes_ltrb = self.dboxes.clone()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5*self.dboxes[:, 2]
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5*self.dboxes[:, 3]
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5*self.dboxes[:, 2]
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5*self.dboxes[:, 3]

    @property
    def scale_xy(self):
        return self.scale_xy_

    @property
    def scale_wh(self):
        return self.scale_wh_

    def __call__(self, order="ltrb"):
        if order == "ltrb": return self.dboxes_ltrb
        if order == "xywh": return self.dboxes


# This class is from https://github.com/kuangliu/pytorch-ssd
class Encoder(object):
    """
        Inspired by https://github.com/kuangliu/pytorch-ssd
        Transform between (bboxes, lables) <-> SSD output

        dboxes: default boxes in size 8732 x 4,
            encoder: input ltrb format, output xywh format
            decoder: input xywh format, output ltrb format

        encode:
            input  : bboxes_in (Tensor nboxes x 4), labels_in (Tensor nboxes)
            output : bboxes_out (Tensor 8732 x 4), labels_out (Tensor 8732)
            criteria : IoU threshold of bboexes

        decode:
            input  : bboxes_in (Tensor 8732 x 4), scores_in (Tensor 8732 x nitems)
            output : bboxes_out (Tensor nboxes x 4), labels_out (Tensor nboxes)
            criteria : IoU threshold of bboexes
            max_output : maximum number of output bboxes
    """

    def __init__(self, dboxes, fast_nms=False):
        self.dboxes = dboxes(order="ltrb")
        self.dboxes_xywh = dboxes(order="xywh").unsqueeze(dim=0)
        self.nboxes = self.dboxes.size(0)
        #print("# Bounding boxes: {}".format(self.nboxes))
        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh
        self.fast_nms = fast_nms

        # self.dboxes = self.dboxes.cuda()
        # self.dboxes_xywh = self.dboxes_xywh.cuda()

    def encode(self, bboxes_in, labels_in, criteria = 0.5):

        try:
            ious = calc_iou_tensor(bboxes_in, self.dboxes)
            best_dbox_ious, best_dbox_idx = ious.max(dim=0)
            best_bbox_ious, best_bbox_idx = ious.max(dim=1)

            # set best ious 2.0
            best_dbox_ious.index_fill_(0, best_bbox_idx, 2.0)

            idx = torch.arange(0, best_bbox_idx.size(0), dtype=torch.int64)
            best_dbox_idx[best_bbox_idx[idx]] = idx

            # filter IoU > 0.5
            masks = best_dbox_ious > criteria
            labels_out = torch.zeros(self.nboxes, dtype=torch.long)
            #print(maxloc.shape, labels_in.shape, labels_out.shape)

            labels_out[masks] = labels_in[best_dbox_idx[masks]]
            bboxes_out = self.dboxes.clone()
            bboxes_out[masks, :] = bboxes_in[best_dbox_idx[masks], :]
            # Transform format to xywh format
            x, y, w, h = 0.5*(bboxes_out[:, 0] + bboxes_out[:, 2]), \
                         0.5*(bboxes_out[:, 1] + bboxes_out[:, 3]), \
                         -bboxes_out[:, 0] + bboxes_out[:, 2], \
                         -bboxes_out[:, 1] + bboxes_out[:, 3]
            bboxes_out[:, 0] = x
            bboxes_out[:, 1] = y
            bboxes_out[:, 2] = w
            bboxes_out[:, 3] = h
        except:
            labels_out = torch.zeros(self.nboxes, dtype=torch.long)
            bboxes_out = torch.zeros(self.nboxes, 4)
        return bboxes_out, labels_out

    def scale_back_batch(self, bboxes_in, scores_in):
        """
            Do scale and transform from xywh to ltrb
            suppose input Nx4xnum_bbox Nxlabel_numxnum_bbox
        """
        if bboxes_in.device == torch.device("cpu"):
            self.dboxes = self.dboxes.cpu()
            self.dboxes_xywh = self.dboxes_xywh.cpu()
        else:
            self.dboxes = self.dboxes.cuda()
            self.dboxes_xywh = self.dboxes_xywh.cuda()
        bboxes_in = bboxes_in.permute(0, 2, 1)
        scores_in = scores_in.permute(0, 2, 1)

        bboxes_in[:, :, :2] = self.scale_xy*bboxes_in[:, :, :2]
        bboxes_in[:, :, 2:] = self.scale_wh*bboxes_in[:, :, 2:]

        bboxes_in[:, :, :2] = bboxes_in[:, :, :2]*self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
        bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp()*self.dboxes_xywh[:, :, 2:]

        # Transform format to ltrb
        l, t, r, b = bboxes_in[:, :, 0] - 0.5*bboxes_in[:, :, 2],\
                     bboxes_in[:, :, 1] - 0.5*bboxes_in[:, :, 3],\
                     bboxes_in[:, :, 0] + 0.5*bboxes_in[:, :, 2],\
                     bboxes_in[:, :, 1] + 0.5*bboxes_in[:, :, 3]

        bboxes_in[:, :, 0] = l
        bboxes_in[:, :, 1] = t
        bboxes_in[:, :, 2] = r
        bboxes_in[:, :, 3] = b

        return bboxes_in, F.softmax(scores_in, dim=-1)

    def decode_batch(self, bboxes_in, scores_in,  criteria=0.45, max_output=200, nms_valid_thresh=0.05):
        bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)

        output = []
        # This split seems dumb to me -- it's already [1, 8732, 4] and [1, 8732, 81]...
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):
            bbox = bbox.squeeze(0)
            prob = prob.squeeze(0)
            if self.fast_nms:
                output.append(self.fast_decode_single(bbox, prob, criteria, max_output,
                             nms_valid_thresh=nms_valid_thresh))
            else:
                try:
                    output.append(self.decode_single(bbox, prob, criteria, max_output,
                                 nms_valid_thresh=nms_valid_thresh))
                except:
                    output.append([
                        torch.Tensor([]).reshape(0, 4).to(bbox.device), \
                        torch.tensor([], dtype=torch.long), \
                        torch.Tensor([]).to(bbox.device)
                    ])
                    return output
            #print(output[-1])
        return output

    # perform non-maximum suppression
    def decode_single(self, bboxes_in, scores_in, criteria, max_output,
                      max_num=200, nms_valid_thresh=0.05):
        # Reference to https://github.com/amdegroot/ssd.pytorch
        bboxes_out = []
        scores_out = []
        labels_out = []

        for i, score in enumerate(scores_in.split(1, 1)):
            # skip background
            # print(score[score>0.90])
            if i == 0: continue
            # print(i)

            score = score.squeeze(1)
            mask = score > nms_valid_thresh

            bboxes, score = bboxes_in[mask, :], score[mask]
            if score.size(0) == 0: continue

            score_sorted, score_idx_sorted = score.sort(dim=0)

            # select max_output indices
            score_idx_sorted = score_idx_sorted[-max_num:]
            candidates = []
            # maxdata, maxloc = scores_in.sort()

            while score_idx_sorted.numel() > 0:
                idx = score_idx_sorted[-1].item()
                bboxes_sorted = bboxes[score_idx_sorted, :]
                bboxes_idx = bboxes[idx, :].unsqueeze(dim=0)
                iou_sorted = calc_iou_tensor(bboxes_sorted,
                                             bboxes_idx).squeeze()
                # we only need iou < criteria
                score_idx_sorted = score_idx_sorted[iou_sorted < criteria]
                candidates.append(idx)

            bboxes_out.append(bboxes[candidates, :])
            scores_out.append(score[candidates])
            labels_out.extend([i] * len(candidates))

        bboxes_out, labels_out, scores_out = torch.cat(bboxes_out, dim=0), \
                                             torch.tensor(labels_out,
                                                          dtype=torch.long), \
                                             torch.cat(scores_out, dim=0)

        _, max_ids = scores_out.sort(dim=0)
        max_ids = max_ids[-max_output:]
        return bboxes_out[max_ids, :], labels_out[max_ids], scores_out[max_ids]

    # perform non-maximum suppression
    def fast_decode_single(self, bboxes_in, scores_in, criteria, max_output, max_num=200, nms_valid_thresh=0.05):
        # Reference to https://github.com/amdegroot/ssd.pytorch
        from SSD import _C as C

        bboxes_out = []
        scores_out = []
        labels_out = []

        # From [8732, num_classes] -> [num_classes, 8732]
        # Makes everything easier.
        scores_in = scores_in.transpose(1, 0)

        # Sort every row (in hopefully a single kernel launch)
        # NOTE: Not masked out things yet
        # NOTE: descending sort is easier to reason about
        # NOTE: Indices are to _global_ bboxes, we're not going to mask them
        score_sorted, score_sorted_idx = scores_in.sort(dim=1, descending=True)

        # Now generate the mask on the sorted scores
        mask = score_sorted > 0.05

        # number of default boxes per class that have a score > 0.05
        splits = mask.sum(dim=1).tolist()

        # only keep scores & indices for default boxes that contribute to this class
        # NOTE: Not masking out bboxes, all indices are global
        score_sorted = score_sorted[mask].split(splits)
        score_sorted_idx = score_sorted_idx[mask].split(splits)

        # assemble prefix sum of splits
        offsets = torch.tensor([0] + list(itertools.accumulate(splits)), dtype=torch.int32, device=bboxes_in.device)

        bboxes_out, scores_out, labels_out = C.nms(1, # N
                                                   scores_in.shape[0],
                                                   offsets,
                                                   torch.cat(score_sorted),
                                                   torch.cat(score_sorted_idx),
                                                   bboxes_in.contiguous(), # VITAL otherwise we get bad results :(
                                                   criteria,
                                                   max_num)

        _, max_ids = scores_out.sort(dim=0)
        max_ids = max_ids[-max_output:]
        return bboxes_out[max_ids, :], labels_out[max_ids], scores_out[max_ids]

def dboxes300_coco():
    figsize = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes

def build_ssd300_coder(fast_nms):
    return Encoder(dboxes300_coco(), fast_nms)
