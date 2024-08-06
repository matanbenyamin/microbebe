import cv2
import numpy as np
from pushbullet import PushBullet
from detectors import BabyDetector
from respiration import RespirationDetector
from matplotlib import pyplot as plt

access_token = 'o.LyVhTiLfvY3HsawYf05RvEVxRs6RDVkL'
pb = PushBullet(access_token)
data = 'microbebe'


bd = BabyDetector(model_path=r'best_head3.onnx')
respiration_detector = RespirationDetector()

cap = cv2.VideoCapture("rtsp://admin:BBJSRM@192.168.1.117:554/ch1/main")
ret, frame = cap.read()
prev_frame = None
iframe = 0
in_bed = False
in_bed_counter = 0
sleep_counter = 0
sleeping = False
motion_counter = 0
wakeup_counter = 0
motion = False
debug = False
measure_resp = False
sent_asleep = False
sent_away = False
sent_in_bed = False

resp_v = []
# init a live respiration graph
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(resp_v)
ax.set_ylim(0, 1)
plt.show()


while True:
    ret, frame = cap.read()
    if not ret:
        continue
    iframe += 1

    #  =====  motion block ===========
    diff_frame = cv2.resize(frame, (100, 100))
    if prev_frame is not None:
        diff = cv2.absdiff(diff_frame, prev_frame)
        diff = cv2.mean(diff)[0]
        if diff > 0.5:
            motion = True
        else:
            motion = False
    prev_frame = diff_frame

    # ======= respiration block ======
    if not motion and measure_resp:
        resp_frame = cv2.resize(frame, (160, 160))
        resp = respiration_detector.process_frame(resp_frame)
        if len(resp_v) > 30:
            if np.mean(resp_v[-30:])  > 0.1:
                breathing = True
            else:
                breathing = False
        resp_v.append(resp)
        if debug:
            line.set_ydata(resp_v)
            line.set_xdata(range(len(resp_v)))
            # reset ylim to last 30 seconds
            ax.set_xlim([np.max([0,len(resp_v) -  30]), len(resp_v)])
            #  ax.autoscale_view()
            plt.draw()
            plt.pause(0.001)


    if iframe % 10 != 0:
        continue

    # detect
    det = bd(frame)
    if det is not None:
        bbox = det[0]
        conf = det[1]

    if conf > 0.6:
        in_bed = True
    else:
        in_bed = False

    # ====== counters ======
    if in_bed:
        in_bed_counter += 1
    else:
        in_bed_counter -= 1
    in_bed_counter = np.min([10,np.max([0, in_bed_counter])])

    if motion:
        motion_counter += 1
    else:
        motion_counter -= 1.5
    motion_counter = np.min([120,np.max([0, motion_counter])])

    if motion_counter < 2 and in_bed_counter > 5:
        sleep_counter += 1
    else:
        sleep_counter -= 1
    sleep_counter = np.min([120,np.max([0, sleep_counter])])

    if in_bed_counter > 5 and sleeping and motion_counter > 50:
        wakeup_counter += 1
    else:
        wakeup_counter -= 1
    wakeup_counter = np.min([10, np.max([0, wakeup_counter])])


    # ===== State machine
    if sleep_counter > 10:
        sleeping = True

    # add the frame number to the frame
    cv2.putText(frame, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if motion_counter > 0:
        cv2.putText(frame, 'MOTION DETECTED', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'{motion_counter}', (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if in_bed_counter > 0:
        cv2.putText(frame, 'IN BED', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'{in_bed_counter}', (350, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('frame', frame)

    # notifications
    if sleeping and not sent_asleep:
        pb.push_note('Baby is sleeping', data)
        sent_asleep = True


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
# close all figs
plt.close('all')