"""
Build pipeline of yolov5s and alphapose, inference one image.

"""

import os
import sys
import cv2
import time
import argparse 
import torchvision
import numpy as np

import torch 
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from detector.yolo_cfg import cfg
from detector.yolov5.utils.general import non_max_suppression

from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.transforms import heatmap_to_coord_simple
from alphapose.utils.pPose_nms import pose_nms

from process_utils import scale_coords, test_transform, plot_pose_res



"""----------------------------- Inference options -----------------------------"""
parser = argparse.ArgumentParser(description='Inference of yolov5s and alphapose')
parser.add_argument('--cfg', 
                    type=str, 
                    default="./configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml", 
                    help='experiment configure file name')
parser.add_argument('--checkpoint', 
                    type=str, 
                    default="./pretrained_models/fast_res50_256x192.pth", 
                    help='checkpoint file name')
parser.add_argument('--inp_path', 
                    type=str, 
                    default="./examples/demo/0.png", 
                    help='the path of input image')
parser.add_argument('--res_path', 
                    type=str, 
                    default="./examples/res/0.jpg", 
                    help='the path of save pose results')
parser.add_argument('--vis_pose', 
                    type=bool, 
                    default=True, 
                    help='whether save the pose result')
parser.add_argument('--inp_size', 
                    type=tuple, 
                    default=(640, 384), 
                    help='the input size of model')

args = parser.parse_args()
cfg = update_config(args.cfg)
args.device = torch.device("cuda:0")
torch.backends.cudnn.benchmark = True
print("Input Size: ", args.inp_size)


def letterbox_image(img, inp_dim=(640, 384)):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas


# Stage1: detection(yolov5s)
def load_det_data(orig_im, inp_path=''):
    
    dim = orig_im.shape[1], orig_im.shape[0]
    # PadResize
    img = (letterbox_image(orig_im, args.inp_size))
    # BGR--->RGB, (640, 640, 3)--->(3, 640, 640)
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    # np.array--->torch.tensor, (0, 255)--->(0, 1), (3, 640, 640)--->(1, 3, 640, 640)
    img = torch.from_numpy(img).float().div(255.).unsqueeze(0)
    im_name = os.path.basename(inp_path)
    return (img, orig_im, im_name, dim)


def load_det_model(opt, cfg):
    weights = "../yolov5/yolov5s.pt"
    from detector.yolov5.models.experimental import attempt_download, attempt_load  # scoped to avoid circular import
    det_model = attempt_load(weights, map_location=None)
    det_model.to(args.device)
    det_model.eval()
    return det_model


def det_forward(det_inps, det_model):
    img, orig_img, im_name, im_dim = det_inps
    img = img.to(args.device)
    with torch.no_grad():
        prediction = det_model(img)
    det_model_out = (prediction, orig_img, im_dim, im_name)
    return det_model_out


# def nms(prediction, conf_thres=0.25, iou_thres=0.45, max_det=100):
#     xc = prediction[..., -2] > conf_thres  # candidates
#     output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    
#     for xi, x in enumerate(prediction):  # image index, image inference
#         x = x[xc[xi]]  # confidence

#         boxes, scores = x[:, :4], x[:, -2]
        
#         i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
#         if i.shape[0] > max_det:  # limit detections
#             i = i[:max_det]

#         output[xi] = x[i]
#     return output

def nms(prediction, conf_thres=0.25, iou_thres=0.45, max_det=100):
    # xc = prediction[..., -2] > conf_thres  # candidates
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    
    for xi, x in enumerate(prediction):  # image index, image inference
        # x = x[xc[xi]]  # confidence

        boxes, scores = x[:, :4], x[:, -2]
        
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]
    return output

"""
def det_post(det_model_out):
    prediction, orig_img, im_shape, im_name = det_model_out

    prediction = nms(prediction)
    prediction = prediction[0] # torch.tensor(num_objs x 6)
    prediction[:, :4] = scale_coords(args.inp_size, prediction[:, :4], im_shape).round()

    dets = prediction.cpu()
    boxes = dets[:, :4]
    
    scores = dets[:, 4:5]
    labels = dets[:, 5:6]
    flag = labels[:, 0] == 1.      # select the person

    ids = torch.zeros(scores.shape)

    inps_idx = torch.zeros(boxes.size(0), 3, 256, 192)
    cropped_boxes = torch.zeros(boxes.size(0), 4)
    det_res = (orig_img, im_name, boxes[flag], scores[flag], ids[flag], inps_idx[flag], cropped_boxes[flag])
    return det_res
"""

def det_post(det_model_out):
    prediction, orig_img, im_shape, im_name = det_model_out
    # prediction = non_max_suppression(prediction, 0.25, 0.45, None, False, max_det=100)
    # prediction = nms(prediction)
    prediction = prediction[0] # torch.tensor(num_objs x 6)
    prediction[:, :4] = scale_coords(args.inp_size, prediction[:, :4], im_shape).round()

    dets = prediction.cpu()
    boxes = dets[:, :4]
    
    scores = dets[:, 4:5]
    # labels = dets[:, 5:6]
    # flag = labels[:, 0] == 1.      # select the person

    ids = torch.zeros(scores.shape)

    inps_idx = torch.zeros(boxes.size(0), 3, 256, 192)
    cropped_boxes = torch.zeros(boxes.size(0), 4)
    # det_res = (orig_img, im_name, boxes[flag], scores[flag], ids[flag], inps_idx[flag], cropped_boxes[flag])
    det_res = (orig_img, im_name, boxes, scores, ids, inps_idx, cropped_boxes)
    
    return det_res


# Stage2: pose estimation(alphapose)
def load_pose_data(det_res):
    with torch.no_grad():
        orig_img, im_name, boxes, scores, ids, inps, cropped_boxes = det_res
        for j, box in enumerate(boxes):
            inps[j], cropped_box = test_transform(orig_img, box)
            cropped_boxes[j] = torch.FloatTensor(cropped_box)

        pose_inps = (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes)
    return pose_inps


def load_pose_model():
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    print('Loading pose model from %s...' % (args.checkpoint,))
    pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    pose_model.to(args.device)
    pose_model.eval()    
    return pose_model 


def pose_forward(pose_inps, pose_model):
    with torch.no_grad():
        (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = pose_inps
        inps = inps.to(args.device)
        hm = pose_model(inps)
        pose_model_out = (boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
    return pose_model_out


def pose_post(pose_model_out):
    boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name = pose_model_out
    hm_data = hm_data.cpu()
    orig_img = np.array(orig_img, dtype=np.uint8)[:, :, ::-1]

    eval_joints = [*range(0,17)]
    hm_size = (64, 48)
    min_box_area = 0

    pose_coords = []
    pose_scores = []
    for i in range(hm_data.shape[0]):
        bbox = cropped_boxes[i].tolist()
        pose_coord, pose_score = heatmap_to_coord_simple(hm_data[i][eval_joints], bbox, hm_shape=hm_size, norm_type=None)
        pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
        pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
    
    preds_img = torch.cat(pose_coords)
    preds_scores = torch.cat(pose_scores)
    boxes, scores, ids, preds_img, preds_scores, pick_ids = \
            pose_nms(boxes, scores, ids, preds_img, preds_scores, min_box_area)

    _result = []
    for k in range(len(scores)):
        _result.append(
            {
                'keypoints':preds_img[k],
                'kp_score':preds_scores[k],
                'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                'idx':ids[k],
                'box':[boxes[k][0], boxes[k][1], boxes[k][2]-boxes[k][0],boxes[k][3]-boxes[k][1]] 
            }
        )

    pose_res = {
        'imgname': im_name,
        'result': _result
    }
    return pose_res 
    

if __name__ == "__main__":
    det_model = load_det_model(args, cfg)
    pose_model = load_pose_model()
    
    img = cv2.imread(args.inp_path)
    det_inps = load_det_data(img)
    print('input size:', det_inps[0].shape)
    det_model_out = det_forward(det_inps, det_model)
    det_res = det_post(det_model_out)

    pose_inps = load_pose_data(det_res)
    pose_model_out = pose_forward(pose_inps, pose_model)
    pose_res = pose_post(pose_model_out)
    if args.vis_pose:
        plot_pose_res(args.inp_path, pose_res, args.res_path)         # validate the pose result
