#!/usr/bin/env python
# coding=utf-8

"""
Composition images as video
"""
import os
import cv2
import argparse
import numpy as np


parser = argparse.ArgumentParser(description="Composition images as video!")
parser.add_argument("--input_dir", 
                    type=str, 
                    default="./pedestrian_720px10", 
                    help="the directory of images")
parser.add_argument("--video_name", 
                    type=str, 
                    default="./pedestrian_720px10.avi", 
                    help="the video path")
args = parser.parse_args()


class CompositionFrames:
    def __init__(self, input_dir, video_name, size=(1280, 720)):
        self.input_dir = input_dir
        self.video_name = video_name
        self.size = size

    def main(self):
        assert os.path.isdir(self.input_dir), f"{self.input_dir} must be an existed directory"
        imgs_name = [x for x in os.listdir(self.input_dir) if x.endswith('jpg')] 
        imgs_name = sorted(imgs_name, key=lambda x: int(x.split('.')[0]))
        imgs_path = [os.path.join(self.input_dir, x) for x in imgs_name] 

        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        videowriter = cv2.VideoWriter(self.video_name, fourcc, 25, self.size)
        
        for img_path in imgs_path[:150]:
            print(img_path)
            img = cv2.imread(img_path)
            videowriter.write(img)

        print('Achieve Writing!')


if __name__ == '__main__':
    composition_frames = CompositionFrames(args.input_dir, args.video_name)
    composition_frames.main()


