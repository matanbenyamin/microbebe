from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import skimage
import numpy as np
import glob
import os
# add the grounding dino to the path
import sys
import matplotlib.pyplot as plt
from torchvision.ops import box_convert
import torch
from PIL import Image
import matplotlib.pyplot as plt


class BabyDetector:
    def __init__(self, model_path=r'/Users/matanb/Downloads/weights/best.pt', conf = 0.001):
        self.model = YOLO(model_path)
        self.conf = conf
        if model_path[-4:] == 'onnx':
            self.onnx = True
        else:
            self.onnx = False

    def __call__(self, frame, policy='smallest'):
        # run inference without printing
        if not self.onnx:
            result = self.model(frame, verbose=False)[0]
            if len(result) == 0:
                # return none with a message of no detection
                return None

            sz = 10000000
            smallest = None
            # tensor to list
            for r in result:
                res = r.boxes.data.tolist()
                raw_roi = res
                roi = [int(x) for x in res[0]]
                roi = [roi[0], roi[1], roi[2] - roi[0], roi[3] - roi[1]]
                if policy == 'smallest':
                    if roi[2] * roi[3] < sz:
                        sz = roi[2] * roi[3]
                        smallest = roi

            self.roi = smallest
        else:
            # if onnx
            roi = self.model.predict(frame, conf=self.conf, imgsz=(320, 320), verbose = False)
            try:
                conf = roi[0].boxes.conf.numpy()[0]
            except:
                conf = 0
            roi = roi[0].boxes.xyxy.data.tolist()
            if len(roi) == 0:
                roi = None
            else:
                roi = [int(x) for x in roi[0]]

            self.roi = roi
            return roi, conf

    def show(self, frame):
        cv2.rectangle(frame, (self.roi[0], self.roi[1]), (self.roi[0]+self.roi[2], self.roi[1]+self.roi[3]), (0, 0, 255), 2)
        plt.imshow(frame)
        plt.show()

bd = BabyDetector(model_path=r'best_head3.onnx')

vid_path = r'C:\Users\lab7\Downloads\2024_8_7_E06290795733_monitoringStoppedNoMeasure_data_1708739012465.mp4'

cap = cv2.VideoCapture(vid_path)
ret, frame = cap.read()

det = bd(frame)

cv2.rectangle(frame, (det[0][0], det[0][1]), (det[0][2], det[0][3]), (0, 0, 255), 2)
plt.imshow(frame)
plt.show()