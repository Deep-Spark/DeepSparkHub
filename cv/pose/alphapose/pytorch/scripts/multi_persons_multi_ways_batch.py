"""
Test the time of pipeline or modules(yolov5s and alphapose) for a single person of multi ways!
In main process of bs = multi ways forward.

"""

import os
import re
import sys
import cv2
import time
import argparse
import torchvision
from torchvision import transforms
import numpy as np

import torch 
import torch.nn as nn
import torch.utils.data as Data

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from detector.yolo_cfg import cfg
from detector.yolov5.utils.general import non_max_suppression

from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.transforms import heatmap_to_coord_simple
from alphapose.utils.pPose_nms import pose_nms

from process_utils import scale_coords, test_transform, plot_pose_res


"""----------------------------- Test Time options -----------------------------"""
parser = argparse.ArgumentParser(description='Test the time of pipeline or modules(yolov5s and alphapose) for a single person of multi ways')
parser.add_argument('--cfg', 
                    type=str, 
                    default="./configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml", 
                    help='experiment configure file name')
parser.add_argument('--checkpoint', 
                    type=str, 
                    default="./pretrained_models/fast_res50_256x192.pth", 
                    help='checkpoint file name')
parser.add_argument('--imgsdir', 
                    type=str, 
                    default="./examples/batch_imgs_multi_persons/singleway", 
                    help='the directory of input image')
parser.add_argument('--flag', 
                    type=str, 
                    default='pipeline', 
                    choices=['pipeline', 'modules'], 
                    help='test the all pipeline or all sub module')
parser.add_argument('--inp_size', 
                    type=tuple, 
                    default=(640, 384), 
                    help='the input size of model')
parser.add_argument('--pose_res', 
                    action='store_true', 
                    help='test the pose results')
parser.add_argument('--FP16', 
                    action='store_true', 
                    help='whether use FP16')
args = parser.parse_args()
cfg = update_config(args.cfg)
args.device = torch.device("cuda:0")
torch.backends.cudnn.benchmark = True
print("Input Size: ", args.inp_size)
if args.FP16:
    print("Forward(yolov5s_alphapose) by FP16")


def cal_avg_time(times_ix):
    assert isinstance(times_ix, list), "The inpust must be a list"
    times = times_ix[:]
    max_index = times.index(max(times))
    del times[max_index]

    min_index = times.index(min(times))
    del times[min_index]
    
    time_avg = sum(times) / len(times)
    return time_avg


def letterbox_image(img, inp_dim=(640, 384)):
    # BGR--->RGB, (384, 640, 3)--->(3, 384, 640)
    img = torch.transpose(img, 1, 2)
    img = torch.transpose(img, 0, 1)
    dst = torch.empty_like(img[0])
    dst.copy_(img[0], non_blocking=True)
    img[0].copy_(img[2], non_blocking=True)
    img[2].copy_(dst, non_blocking=True)
    '''resize image with unchanged aspect ratio using padding'''
    transform = transforms.Compose([transforms.Resize(size=(360, 640)), 
                                    transforms.Pad(padding=(0, 12), fill=128)])
    return transform(img)


def load_det_model(opt, cfg):
    weights = "../yolov5/yolov5s.pt"
    from detector.yolov5.models.experimental import attempt_download, attempt_load  # scoped to avoid circular import
    det_model = attempt_load(weights, map_location=None)
    if args.FP16:
        det_model.half()
    det_model.to(args.device)
    det_model.eval()
    return det_model


def batched_nms(prediction, conf_thres=0.25, iou_thres=0.45, max_det=100):
    output = []
    bs, boxes_num, _ = prediction.shape
    idxs = torch.from_numpy(np.arange(bs)).unsqueeze(-1).expand(bs, boxes_num)

    xc = prediction[..., 4] > conf_thres  # candidates
    
    x = prediction[xc]  # confidence
    idxs = idxs[xc]

    boxes, scores = x[:, :4], x[:, 4]
    i = torchvision.ops.batched_nms(boxes, scores, idxs, iou_thres)  # NMS
    
    res = x[i]
    idxs_batch = idxs[i]
    for j in range(bs):
        output.append(res[idxs_batch == j])
    return output


def batched_det_post_to_pose_data(prediction, orig_img, im_dim, im_name):
    pose_inps = []
    pose_infor = []
    pose_index = []
    
    for i, (pred, y, z) in enumerate(zip(prediction, orig_img, im_name)):
        pred[:, :4] = scale_coords(args.inp_size, pred[:, :4], (1280, 720)).round()
        det = pred.cpu()
        
        boxes = det[:, :4]
        scores = det[:, 4:5]
        labels = det[:, 5:6]
        flag = labels[:, 0] == 1.      # select the person

        ids = torch.zeros(scores.shape)

        inps_idx = torch.zeros(boxes.size(0), 3, 256, 192)
        cropped_boxes = torch.zeros(boxes.size(0), 4)
        # det_res = (y, z, boxes[flag], scores[flag], ids[flag], inps_idx[flag], cropped_boxes[flag])
        inps = inps_idx[flag]
        for j, box in enumerate(boxes[flag]):
            inps[j], cropped_box = test_transform(y, box)
            cropped_boxes[j] = torch.FloatTensor(cropped_box)
        pose_infor.append((y, z, boxes[flag], scores[flag], ids[flag], cropped_boxes[flag]))
        pose_inps.append(inps)
        pose_index.extend([i for j in range(len(boxes[flag]))])

    pose_inps = torch.cat(pose_inps, dim=0)
    pose_index = torch.tensor(pose_index)
    return pose_inps, pose_infor, pose_index


def detection(data, det_model):
    img, orig_img, im_name, im_dim = data
    with torch.no_grad():
        prediction = det_model(img)
    
    prediction = batched_nms(prediction)
    pose_inps, pose_infor, pose_index = batched_det_post_to_pose_data(prediction, orig_img, im_dim, im_name)

    return pose_inps, pose_infor, pose_index


# Stage2: pose estimation(alphapose)
def load_pose_model():
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    print('Loading pose model from %s...' % (args.checkpoint,))
    pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    if args.FP16:
        pose_model.half()
    pose_model.to(args.device)
    pose_model.eval()    
    return pose_model 


def pose_post(hm_data, pose_infor):
    orig_img, im_name, boxes, scores, ids, cropped_boxes = pose_infor
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

    result = {
        'imgname': im_name,
        'result': _result
    }
    return result 


def pose(pose_inps, pose_infor, pose_model, pose_index):
    pose_out = []
    with torch.no_grad():
        pose_inps = pose_inps.to(args.device)
        if args.FP16:
            pose_inps = pose_inps.half()
        hm = pose_model(pose_inps)
    
    if args.FP16:
        hm = hm.float()
    
    for i, infor in enumerate(pose_infor):
        hm_data = hm[pose_index == i]
        if hm_data.dim == 3:
            hm_data = hm_data.unsqeeze(0)
        res = pose_post(hm_data, infor)
        pose_out.append(res)
    return pose_out
 

def preprocess(orig_im):
    # PadResize
    img = (letterbox_image(orig_im, args.inp_size))
    #(0, 255)--->(0, 1), (3, 384, 640)--->(1, 3, 384, 640)
    img = img.div(255.).unsqueeze(0)
    return img


stream_length = 4
streams = [torch.cuda.Stream() for i in range(stream_length)]


def all_pipeline_time(det_model, pose_model):
    times = []
    for i in range(256):
        s = time.time()
        
        rootdir = args.imgsdir
        imgnames = [x for x in os.listdir(rootdir) if x.endswith('png') or x.endswith('jpg')]
        imgpaths = [os.path.join(rootdir, imgname) for imgname in imgnames]    
        orig_im_batch = [cv2.imread(imgpath) for imgpath in imgpaths]
        
        s = time.time()
        dims_batch = [(orig_im.shape[1], orig_im.shape[0]) for orig_im in orig_im_batch]
        img_batch = []
        for idx, orig_im in enumerate(orig_im_batch):
            with torch.cuda.stream(streams[idx % stream_length]):
                if args.FP16:
                    img_batch.append(preprocess(torch.from_numpy(orig_im).pin_memory().to(args.device, non_blocking=True).to(torch.half)))
                else:
                    img_batch.append(preprocess(torch.from_numpy(orig_im).pin_memory().to(args.device, non_blocking=True).to(torch.float)))

        torch.cuda.synchronize()
        
        img = torch.cat(img_batch, dim=0)
        data = (img, orig_im_batch, imgnames, dims_batch)
        
        pose_inps, pose_infor, pose_index = detection(data, det_model)
        pose_out = pose(pose_inps, pose_infor, pose_model, pose_index)
        e = time.time()
        time_once = e - s
        times.append(time_once)
        print('iter: %d, time: %.6f' % (i, time_once))
    
    avg_times = cal_avg_time(times)
    print('avg_times: %.6f' % (avg_times))
    

def all_module_time(det_model, pose_model):
    times_ix = []
    prof_load_det_data = []
    prof_det_forward = []
    prof_det_post = []
    prof_load_pose_data = []
    prof_pose_forward = []
    prof_pose_post = []
    
    for i in range(256):
        rootdir = args.imgsdir
        imgnames = [x for x in os.listdir(rootdir) if x.endswith('png') or x.endswith('jpg')]
        imgpaths = [os.path.join(rootdir, imgname) for imgname in imgnames]    

        orig_im_batch = [cv2.imread(imgpath) for imgpath in imgpaths]
        
        t0 = time.time()
        dims_batch = [(orig_im.shape[1], orig_im.shape[0]) for orig_im in orig_im_batch]
        img_batch = []
        for idx,orig_im in enumerate(orig_im_batch):
            with torch.cuda.stream(streams[idx%stream_length]):
                if args.FP16:
                    img_batch.append(preprocess(torch.from_numpy(orig_im).pin_memory().to(args.device, non_blocking=True).to(torch.half)))
                else:
                    img_batch.append(preprocess(torch.from_numpy(orig_im).pin_memory().to(args.device, non_blocking=True).to(torch.float)))

        torch.cuda.synchronize()

        img = torch.cat(img_batch, dim=0)
        data = (img, orig_im_batch, imgnames, dims_batch)
        torch.cuda.synchronize()
        t1 = time.time()
        prof_load_det_data.append(t1 - t0)
        
        img, orig_img, im_name, im_dim = data
        with torch.no_grad():
            prediction = det_model(img)
        torch.cuda.synchronize()
        t2 = time.time()
        prof_det_forward.append(t2 - t1)
        
        det_out = []
        prediction = batched_nms(prediction)
        pose_inps, pose_infor, pose_index = batched_det_post_to_pose_data(prediction, orig_img, im_dim, im_name)
        t3 = time.time()
        prof_det_post.append(t3 - t2)
        
        torch.cuda.synchronize()
        t4 = time.time()
        prof_load_pose_data.append(t4 - t3)
        
        pose_out = []
        with torch.no_grad():
            pose_inps = pose_inps.to(args.device)
            if args.FP16:
                pose_inps = pose_inps.half()
            hm = pose_model(pose_inps)
        torch.cuda.synchronize()
        t5 = time.time()
        prof_pose_forward.append(t5 - t4)
        
        if args.FP16:
            hm = hm.float()
        
        for i, infor in enumerate(pose_infor):
            hm_data = hm[pose_index == i]
            if hm_data.dim == 3:
                hm_data = hm_data.unsqeeze(0)
            res = pose_post(hm_data, infor)
            pose_out.append(res)
        
        torch.cuda.synchronize()
        t6 = time.time()
        prof_pose_post.append(t6 - t5)
        times_ix.append((t6 - t0))
    
    time_avg = cal_avg_time(times_ix)
    t1_avg = cal_avg_time(prof_load_det_data)
    t2_avg = cal_avg_time(prof_det_forward)
    t3_avg = cal_avg_time(prof_det_post)
    t4_avg = cal_avg_time(prof_load_pose_data)
    t5_avg = cal_avg_time(prof_pose_forward)
    t6_avg = cal_avg_time(prof_pose_post)
    
    print("""
                *************************************************
                pipeline_time_avg:       {:.6f} s,
                prof_load_det_data_avg:  {:.6f} s,
                prof_det_forward_avg:    {:.6f} s,
                prof_det_post_avg:       {:.6f} s,
                prof_load_pose_data_avg: {:.6f} s, 
                prof_pose_forward_avg:   {:.6f} s, 
                prof_pose_post_avg:      {:.6f} s, 
        """.format(time_avg, t1_avg, t2_avg, t3_avg,  t4_avg, t5_avg, t6_avg))


def test_pose(det_model, pose_model):
    rootdir = './examples/batch_imgs_multi_persons/singleway'
    imgnames = [x for x in os.listdir(rootdir) if x.endswith('png') or x.endswith('jpg')]
    imgpaths = [os.path.join(rootdir, imgname) for imgname in imgnames]    

    orig_im_batch = [cv2.imread(imgpath) for imgpath in imgpaths]
    
    dims_batch = [(orig_im.shape[1], orig_im.shape[0]) for orig_im in orig_im_batch]
    img_batch = [preprocess(torch.from_numpy(orig_im).pin_memory().to(args.device, non_blocking=True)) for orig_im in orig_im_batch]
    
    img = torch.cat(img_batch, dim=0)
    if args.FP16:
        img = img.half()
    data = (img, orig_im_batch, imgnames, dims_batch)
    
    pose_inps, pose_infor, pose_index = detection(data, det_model)
    pose_out = pose(pose_inps, pose_infor, pose_model, pose_index)
    
    plot_pose_res(imgpaths[0], pose_out[0], './res.jpg')

    
if __name__ == "__main__":
    det_model = load_det_model(args, cfg)
    pose_model = load_pose_model()
    
    if args.pose_res:
        test_pose(det_model, pose_model)
    else:
        if args.flag == 'pipeline':
            call_func = all_pipeline_time      # test the time of all pipeline 
        else:  
            call_func = all_module_time        # test the time of six module time
    
        results = call_func(det_model, pose_model) 
