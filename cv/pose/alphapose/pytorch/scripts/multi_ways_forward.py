"""
Test the time of pipeline or modules(yolov5s and alphapose) for a single person of multi ways!
Contain multi process of bs = 1 forward.

"""

import os
import sys
import cv2
import time
import argparse
import torchvision
import numpy as np
import multiprocessing 
from torch.multiprocessing import Process
from threading import Thread
import re

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
                    default="./examples/test_stream_videos.txt", 
                    help='the path of input image')
parser.add_argument('--flag', 
                    type=str, 
                    default='pipeline', 
                    choices=['pipeline', 'modules'], 
                    help='test the all pipeline or all sub module')
parser.add_argument('--inp_size', 
                    type=tuple, 
                    default=(640, 384), 
                    help='the input size of model')

args = parser.parse_args()
cfg = update_config(args.cfg)
args.device = torch.device("cuda:0")
torch.backends.cudnn.benchmark = True
print("Input Size: ", args.inp_size)


# load stream
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


def load_det_data(orig_im, inp_path=''):
    dim = orig_im.shape[1], orig_im.shape[0]
    # PadResize
    img = (letterbox_image(orig_im, args.inp_size))
    # BGR--->RGB, (384, 640, 3)--->(3, 384, 640)
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    # np.array--->torch.tensor, (0, 255)--->(0, 1), (3, 384, 640)--->(1, 3, 384, 640)
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


def det_forward(inps, det_model):
    img, orig_img, im_name, im_dim = inps
    img = img.to(args.device)
    with torch.no_grad():
        prediction = det_model(img)
    det_model_out = (prediction, orig_img, im_dim, im_name)
    return det_model_out


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
    
    # prediction = non_max_suppression(prediction, 0.25, 0.45, None, False, max_det=100)
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

    result = {
        'imgname': im_name,
        'result': _result
    }
    return result 
 

def all_pipeline_time(inp_path, det_model, pose_model, q):
    dataset = LoadStreams(inp_path)

    times_ix = []
    for frame_cnt, img in dataset:
        if img[0] is None or frame_cnt == 256:
            break
        torch.cuda.synchronize()
        s = time.time()
        
        inps = load_det_data(img[0])
        det_model_out = det_forward(inps, det_model)
        det_res = det_post(det_model_out)
        pose_inps = load_pose_data(det_res)
        if pose_inps[0].shape[0] == 0:
            continue
        pose_model_out = pose_forward(pose_inps, pose_model)
        pose_res = pose_post(pose_model_out)
        torch.cuda.synchronize()
        e = time.time()
        times_ix.append((e - s))

        if frame_cnt % 10 == 0:
            print('iter: %d, %.6f ms' % (frame_cnt, (e - s) * 1E3))

    time_avg = cal_avg_time(times_ix)
    
    q.put(time_avg)


def all_module_time(inp_path, det_model, pose_model, q):
    dataset = LoadStreams(inp_path)

    times_ix = []
    prof_load_det_data = []
    prof_det_forward = []
    prof_det_post = []
    prof_load_pose_data = []
    prof_pose_forward = []
    prof_pose_post = []
    for frame_cnt, img in dataset:
        if img[0] is None or frame_cnt == 256:
            break
        torch.cuda.synchronize()
        t0 = time.time()
        
        inps = load_det_data(img[0])
        torch.cuda.synchronize()
        t1 = time.time()
        prof_load_det_data.append(t1 - t0)

        det_model_out = det_forward(inps, det_model)
        torch.cuda.synchronize()
        t2 = time.time()
        prof_det_forward.append(t2 - t1)

        det_res = det_post(det_model_out)
        torch.cuda.synchronize()
        t3 = time.time()
        prof_det_post.append(t3 - t2)

        pose_inps = load_pose_data(det_res)
        torch.cuda.synchronize()
        t4 = time.time()
        prof_load_pose_data.append(t4 - t3)

        if pose_inps[0].shape[0] == 0:
            continue
        
        pose_model_out = pose_forward(pose_inps, pose_model)
        torch.cuda.synchronize()
        t5 = time.time()
        prof_pose_forward.append(t5 - t4)

        pose_res = pose_post(pose_model_out)
        torch.cuda.synchronize()
        t6 = time.time()
        prof_pose_post.append(t6 - t5)

        times_ix.append((t6 - t0))

        if frame_cnt % 10 == 0:
            print('iter: %d, %.6f ms' % (frame_cnt, (t6 - t0) * 1E3))

    time_avg = cal_avg_time(times_ix)
    prof_load_det_data_avg = cal_avg_time(prof_load_det_data)
    prof_det_forward_avg = cal_avg_time(prof_det_forward)
    prof_det_post_avg = cal_avg_time(prof_det_post)
    prof_load_pose_data_avg = cal_avg_time(prof_load_pose_data)
    prof_pose_forward_avg = cal_avg_time(prof_pose_forward)
    prof_pose_post_avg = cal_avg_time(prof_pose_post)
    
    q.put([time_avg, prof_load_det_data_avg, prof_det_forward_avg, prof_det_post_avg, prof_load_pose_data_avg, prof_pose_forward_avg, prof_pose_post_avg])
        

if __name__ == "__main__":

    torch.multiprocessing.set_start_method('spawn', force=True)
    with open(args.inp_path, 'r') as f:
        inp_path_list = f.read().splitlines()
    
    n = len(inp_path_list)
    assert n > 0
    print('num process:', n)
    all_pipelines = []
    all_pipelines_q = torch.multiprocessing.Queue()
    
    if args.flag == 'pipeline':
        call_func = all_pipeline_time      # test the time of all pipeline 
    else:  
        call_func = all_module_time        # test the time of six module time
    
    # the number of process according the videos txt file
    for i in range(n):
        det_model = load_det_model(args, cfg)
        pose_model = load_pose_model()
        p = Process(target=call_func, args=(inp_path_list[i], det_model, pose_model, all_pipelines_q)) 
        all_pipelines.append(p) 
    
    for p in all_pipelines:
        p.start()
    
    
    if args.flag == 'pipeline':
        results = [all_pipelines_q.get() for p in all_pipelines]
        print('pipeline_num: {}, pipeline_time: {}, pipeline_time_avg: {:.6f}'.format(n, results, sum(results) / len(results)))
    else: 
        results = np.array([all_pipelines_q.get() for p in all_pipelines])
        avg_results = tuple(np.sum(results, axis=0) / n)

        print("""
                *************************************************
                pipeline_num:            {}, 
                pipeline_time:           {}, 
                pipeline_time_avg:       {:.6f} ms,
                prof_load_det_data_avg:  {:.6f} ms,
                prof_det_forward_avg:    {:.6f} ms,
                prof_det_post_avg:       {:.6f} ms,
                prof_load_pose_data_avg: {:.6f} ms, 
                prof_pose_forward_avg:   {:.6f} ms, 
                prof_pose_post_avg:      {:.6f} ms, 
                sum_prof:                {:.6f} ms
                """
                .format(n, results, *avg_results, sum(avg_results)-avg_results[0]))
