import cv2
import time
import threading

import numpy as np
from ultralytics import YOLO

# ====  flags
in_bed = False
in_bed_counter = 0
sleeping = False
sleep_counter = 0
motion = False
motion_counter = 0

model = YOLO('yolov10n.pt')

# === Params
T_det = 0.5
T_sleep_min = 3
T_awake_min = 2
T_viz = 0.1

# === inits
prev_frame = None
frame = None
detection_results = None
stop_event = threading.Event()

cap = cv2.VideoCapture("http://192.168.1.106:8080/video")
# grab strategy last frame
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

def detection_thread(stop_event):
    global detection_results, frame, model, T_det
    while not stop_event.is_set():
        if frame is not None:
            start_time = time.time()
            results = model(frame)
            detection_results = results[0]
            elapsed_time = time.time() - start_time
            if elapsed_time < T_det:
                time.sleep(T_det - elapsed_time)

# Start the detection thread
thread = threading.Thread(target=detection_thread, args=(stop_event,))
thread.daemon = True
thread.start()

# ======== Main loop =============
t_viz = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    diff_frame = cv2.resize(frame, (20,20))
    if prev_frame is not None:
        diff = cv2.absdiff(diff_frame, prev_frame)
        diff = np.mean(diff)
        if diff > 0:
            print(diff)
        if diff > 0.5:
            motion = True
            motion_counter = 0
        else:
            motion = False
            motion_counter += 1
        if motion_counter > 10:
            motion = False

    # ======== Detection =============
    if detection_results is not None:
        for r in detection_results.boxes:
            if r.cls == 0:
                box = r.data
                # tensor to list
                box = box.tolist()[0]
                box = [int(b) for b in box]
                # Draw bounding box
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    if time.time() - t_viz > T_viz:
        cv2.imshow('frame', frame)
        t_viz = time.time()

    prev_frame = diff_frame.copy()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Signal the detection thread to stop and wait for it to finish
stop_event.set()
thread.join()

cap.release()
cv2.destroyAllWindows()
