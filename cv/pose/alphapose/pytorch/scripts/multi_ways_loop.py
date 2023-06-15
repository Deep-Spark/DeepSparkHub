"""
Test the time of pipeline or modules(yolov5s and alphapose) for a single person of multi ways!
single process:  process data by loop
single process: model forward(batch_size = the number of loop for process data)
"""

import os
import re
import sys
import cv2
import time
import argparse
import torchvision
import numpy as np
import multiprocessing 
# from multiprocessing import Process
from threading import Thread
from torch.multiprocessing import Process

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

from process_utils import scale_coords, test_transform


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
parser.add_argument('--inp_path', 
                    type=str, 
                    default="./examples/test_stream_jpg.txt", 
                    help='the path of input image')
parser.add_argument('--flag', 
                    type=str,
                    default='pipeline',
                    help='test all pipeline or all parts')
parser.add_argument('--inp_size', 
                    type=tuple, 
                    default=(640, 384), 
                    help='the input size of model')
args = parser.parse_args()
cfg = update_config(args.cfg)
args.device = torch.device("cuda:0")
torch.backends.cudnn.benchmark = True
print("Input Size: ", args.inp_size)


# ================================Step1 Load Video Stream================================
def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


class LoadStreams:
    # YOLOv5 streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    def __init__(self, sources='streams.txt', img_size=640, stride=32, auto=True):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources) as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.auto = auto
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f'{i + 1}/{n}: {s}... '
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'{st}Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.fps[i] = 25#max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0  # 30 FPS fallback
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback

            _, self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            print(f"{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        print('')  # newline

        # check for common shapes
        #print([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0].shape for x in self.imgs])
        #s = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0].shape for x in self.imgs])
        #self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        #if not self.rect:
        #    print('WARNING: Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap, stream):
        # Read stream `i` frames in daemon thread
        n, f, read = 0, self.frames[i], 1  # frame number, frame array, inference every 'read' frame
        while cap.isOpened() and n < f:
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n % read == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    print('WARNING: Video stream unresponsive, please check your IP camera connection.')
                    self.imgs[i] *= 0
                    cap.open(stream)  # re-open stream if signal was lost
            time.sleep(1 / self.fps[i])  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        _frame_cnt = self.count
        img0 = self.imgs.copy()

        return _frame_cnt, img0

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


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
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas


# ================================Step2 Yolov5s Forward===========================
def load_det_data(orig_im, inp_path):
    dim = orig_im.shape[1], orig_im.shape[0]
    # PadResize
    img = (letterbox_image(orig_im, args.inp_size))
    # BGR--->RGB, (384, 640, 3)--->(3, 384, 640)
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    # np.array--->torch.tensor, (0, 255)--->(0, 1), (3, 384, 640)--->(1, 3, 384, 640)
    img = torch.from_numpy(img).float().div(255.).unsqueeze(0)
    im_name = os.path.basename(inp_path)
    return (img, orig_im, dim, im_name)


def load_det_model(opt, cfg):
    weights = "../yolov5/yolov5s.pt"
    from detector.yolov5.models.experimental import attempt_download, attempt_load  # scoped to avoid circular import
    det_model = attempt_load(weights, map_location=None)
    det_model.to(args.device)
    det_model.eval()
    return det_model


def nms(prediction, conf_thres=0.25, iou_thres=0.45, max_det=100):
    xc = prediction[..., 4] > conf_thres  # candidates
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence
        
        boxes, scores = x[:, :4], x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]
    return output


def det_post(det_model_out):
    prediction, orig_img, im_shape, im_name = det_model_out
    # prediction = non_max_suppression(prediction, 0.5, 0.5, None, False, max_det=100)    # list 
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


# ================================Step3 Alphapose Forward===========================
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


def pose_post(pose_model_out):
    hm, orig_img, im_name, boxes, scores, ids, cropped_boxes = pose_model_out
    hm_data = hm.cpu()
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


def run():
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    det_model = load_det_model(args, cfg)
    pose_model = load_pose_model()
    
    with open(args.inp_path, 'r') as f:
        inp_path_list = f.read().splitlines()
    
    n = len(inp_path_list)
    assert n > 0
    
    # stage1 loop load data(detection)
    det_inps0 = []
    for i in range(n):
        img = cv2.imread(inp_path_list[i])
        det_inps_idx = load_det_data(img, '')
        det_inps0.append(det_inps_idx)

    det_inps = [x[0] for x in det_inps0]
    det_inps = torch.cat(det_inps, dim=0)   
    
    # stage2 yolov5s forward
    with torch.no_grad():
        det_inps = det_inps.to(args.device)
        det_out = det_model(det_inps)

    # stage3 loop post process(detection)
    det_res = []
    for i in range(n):
        det_res_idx = det_post((det_out[i:i+1],) + det_inps0[i][1:])
        det_res.append(det_res_idx)
    
    # stage4 loop load data(pose)
    pose_inps0 = [load_pose_data(det_res[i]) for i in range(n)]
    pose_inps = [x[0] for x in pose_inps0]
    pose_inps = torch.cat(pose_inps, dim=0)

    # stage5 alphapose forward
    with torch.no_grad():
        pose_inps = pose_inps.to(args.device)
        pose_model_out = pose_model(pose_inps)
    
    pose_res = [pose_post((pose_model_out[i:i+1], ) + pose_inps0[i][1:]) for i in range(n)]
    

if __name__ == '__main__':
    run()
