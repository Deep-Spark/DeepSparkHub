#!/usr/bin/env python
# coding=utf-8

import os
import cv2 
import numpy as np 
import torch


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = cv2.imread(img)
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    # pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    pad = (img1_shape[0] - img0_shape[0] * gain) / 2, (img1_shape[1] - img0_shape[1] * gain) / 2# wh padding

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    return coords

def scale_coords_(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        # pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        pad = (img1_shape[0] - img0_shape[0] * gain) / 2, (img1_shape[1] - img0_shape[1] * gain) / 2# wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def test_transform(src, bbox):
    from alphapose.utils.bbox import _box_to_center_scale, _center_scale_to_box
    from alphapose.utils.transforms import get_affine_transform, im_to_torch

    aspect_ratio = float(192 / 256)
    input_size = (192, 256)
    xmin, ymin, xmax, ymax = bbox
    center, scale = _box_to_center_scale(
        xmin, ymin, xmax - xmin, ymax - ymin, aspect_ratio)
    scale = scale * 1.0

    inp_w, inp_h = input_size

    trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
    img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
    bbox = _center_scale_to_box(center, scale)

    img = im_to_torch(img)
    img[0].add_(-0.406)
    img[1].add_(-0.457)
    img[2].add_(-0.480)

    return img, bbox


def plot_pose_res(inp_path, pose_res, res_path):
    """
    Plot the pose and box in the image 
    Arguments:
        inp_path: (type: str)
        pose_res: (type: dict) 
        res_path: (type: str)
    """
    im_name = pose_res['imgname']
    result = pose_res['result']
    
    img = cv2.imread(inp_path)
    for temp in result:
        keypoints = temp['keypoints'].numpy().tolist()
        for keypoint in keypoints:
            keypoint = [int(x) for x in keypoint]
            x, y = keypoint
            cv2.circle(img, (x, y), 2, (0, 255, 0), 2)

        box = temp['box']
        box = [int(x) for x in box]
        x, y, w, h = box 
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imwrite(res_path, img)
    print(f"Save pose result image in {res_path}")
