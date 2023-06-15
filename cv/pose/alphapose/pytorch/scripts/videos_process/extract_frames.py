#!/usr/bin/env python
# coding=utf-8

"""
Extract frames from video
"""
import os
import cv2
import argparse
import numpy as np


parser = argparse.ArgumentParser(description="Extract frames from test videos!")
parser.add_argument("--input_dir", 
                    type=str, 
                    default="/home/datasets/cv/testdata/videos", 
                    help="the directory of videos")
parser.add_argument("--result_dir", 
                    type=str, 
                    default="/home/datasets/cv/testdata/images", 
                    help="the saved directory of frames")
args = parser.parse_args()


class ExtractFrames:
    def __init__(self, input_dir, result_dir):
        self.input_dir = input_dir
        self.result_dir = result_dir

    def _parse_input(self):
        assert os.path.isdir(self.input_dir), f"{self.input_dir} must be an existed directory"
        videos_name = [x for x in os.listdir(self.input_dir) if x.endswith('avi')] 
        videos_path = [os.path.join(self.input_dir, x) for x in videos_name] 
        return videos_path 

    def _extract_frame(self, video_path):
        assert os.path.isfile(video_path), f"{video_path} must be an existed file"
        video_name = os.path.basename(video_path).split('.')[0]
        print(video_name)

        images_dir = os.path.join(self.result_dir, video_name)
        if not os.path.exists(images_dir):
            os.mkdir(images_dir)

        video_capture = cv2.VideoCapture()
        video_capture.open(video_path)

        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print("fps=", fps, "; frames=", frames)

        for i in range(frames):
            if i % 100 == 0:
                print('Process: %d / %d' % (i, frames))
            ret, frame = video_capture.read()
            try:
                cv2.imwrite(os.path.join(images_dir, "%d.jpg" % i), frame)           
            except:
                break

    def main(self):
        videos_path = self._parse_input()

        for video_path in videos_path:
            self._extract_frame(video_path)


if __name__ == '__main__':
    extract_frames = ExtractFrames(args.input_dir, args.result_dir)
    extract_frames.main()
