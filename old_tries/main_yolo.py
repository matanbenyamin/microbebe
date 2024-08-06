from pushbullet import PushBullet
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
import logging
from ultralytics import YOLO
access_token = 'o.LyVhTiLfvY3HsawYf05RvEVxRs6RDVkL'
pb = PushBullet(access_token)
data = 'microbebe'


yolo = YOLO('yolov10m.pt')

# opt
cap = cv2.VideoCapture("rtsp://admin:BBJSRM@192.168.1.117:554/ch1/main")

prev_frame = None
motion_counter = 0
T_det = 0.5
patience = 50
bbox = None
sleep_counter = 0
in_bed = False
once_in_bed = False
in_bed_counter = 0
sent_asleep = False
sent_awake = False
sent_in_bed = False
motion_counter_threshold = 89
num_stragiht_detections = 0
# The detector is initialized. Use it here.
t0 = time.time()
t0_log = time.time()
i_frame = 0
while True:
    ret, frame = cap.read()
    i_frame += 1
    # frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    if not ret:
        continue
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    timestamp = time.time()
    # detector.detect_async(mp_image, int(timestamp * 1e6))
    if time.time() - t0 > T_det:
        detection_result = yolo(frame)
        t0 = time.time()
        if len(detection_result[0].boxes) > 0:
            for idx, r in enumerate(detection_result[0].boxes):
                box = r.xyxy
                cat = detection_result[0].names[r.cls.numpy()[0]]
                box = box.tolist()[0]
                box = [int(b) for b in box]
                bbox = box
                score = r.conf.numpy()[0]
                break
            if score > 0.5 and cat == 'person':
                num_stragiht_detections += 1
                if num_stragiht_detections > 5:
                    in_bed = True
                    once_in_bed = True
                    in_bed_counter = 0
            else:
                score = None
                bbox = None
                num_stragiht_detections = 0
                if once_in_bed:
                    in_bed_counter += 1
                if in_bed_counter > patience:
                    in_bed = False
                    once_in_bed = False
                    in_bed_counter = 0
                    num_stragiht_detections = 0

        else:
            cat = None
            score = None
            bbox = None
            num_stragiht_detections = 0

            if once_in_bed:
                in_bed_counter += 1
            if in_bed_counter > patience:
                in_bed = False
                once_in_bed = False
                in_bed_counter = 0
    if bbox:
        start_point = bbox[0], bbox[1]
        end_point = bbox[2], bbox[3]
        cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 3)
        cv2.putText(frame, f'{cat} ({score})', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 255, 0), 1)

    if i_frame % 10 == 0:
        if prev_frame is not None:
            if bbox:
                norm_frame = cv2.resize(
                    frame[bbox[1]:bbox[3], bbox[0]:bbox[2]],
                    (20, 20))
                norm_prev_frame = cv2.resize(
                    prev_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]],
                    (20, 20))
                diff = cv2.absdiff(norm_frame, norm_prev_frame)
                diff = np.mean(diff)
                if diff < 10:
                    motion_counter += 10

                motion_counter *= 0.9
                motion_counter = min(motion_counter, motion_counter_threshold+1)


        prev_frame = frame


    # route analyzer
    if motion_counter > 89.9999999 and not sent_asleep:
        push = pb.push_note(data, 'Ella is sleeping')
        sent_asleep = True
        sent_awake = False
    if sent_asleep and motion_counter < 4 and not sent_awake:
        push = pb.push_note(data, 'Ella is waking up')
        sent_awake = True
        sent_asleep = False
    if in_bed and not sent_in_bed:
        push = pb.push_note(data, 'Ella is in bed')
        sent_in_bed = True
    if not in_bed and sent_in_bed:
        push = pb.push_note(data, 'Ella is out of bed')
        sent_in_bed = False

    cv2.imshow('frame', frame)

    # build log row to print once a second
    if time.time() - t0_log > 5:
        t0_log = time.time()
        logging.info(f'{cat} ({score})')
        logging.info(f'motion counter: {motion_counter}')
        logging.info(f'in bed: {in_bed}')
        logging.info(f'sent asleep: {sent_asleep}')
        logging.info(f'sent awake: {sent_awake}')
        logging.info(f'sent in bed: {sent_in_bed}')
        logging.info(f'once in bed: {once_in_bed}')
        logging.info(f'in bed counter: {in_bed_counter}')
        logging.info(f'num straight detections: {num_stragiht_detections}')
        logging.info(f'bbox: {bbox}')

        # print log row in one long print no line breaks
        print(f'{cat} ({score})', end=' ')
        print(f'motion counter: {motion_counter}', end=' ')
        print(f'in bed: {in_bed}', end=' ')
        print(f'sent asleep: {sent_asleep}', end=' ')
        print(f'sent awake: {sent_awake}', end=' ')
        print(f'sent in bed: {sent_in_bed}', end=' ')
        print(f'once in bed: {once_in_bed}', end=' ')
        print(f'in bed counter: {in_bed_counter}', end=' ')
        print(f'num straight detections: {num_stragiht_detections}')




    if cv2.waitKey(1) & 0xFF == ord('q'):
        break