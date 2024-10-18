import cv2
import datetime
cap = cv2.VideoCapture("http://192.168.1.106:8080/video")


# collect 1000 frames, one after every signifcant motion
prev_frame = None
N = 1000
n=0
while n < N:
    ret, frame = cap.read()
    if not ret:
        break
    if prev_frame is not None:
        diff = cv2.absdiff(prev_frame, frame)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        if cv2.countNonZero(diff) > 10000:
            # filename is date and time
            filename = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f') + f'frame_{n}.jpg'
            # save to data folder
            cv2.imwrite('data/' + filename, frame)
            n += 1
    prev_frame = frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
