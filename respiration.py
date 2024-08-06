import cv2
import numpy as np

class RespirationDetector:
    def __init__(self):
        self.previous_frame = None
        self.flow_waveform = []

    def process_frame(self, frame):
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Initialize previous_frame if it's None
        if self.previous_frame is None:
            self.previous_frame = gray_frame
            return None

        # Compute optical flow between the previous frame and the current frame
        flow = cv2.calcOpticalFlowFarneback(self.previous_frame, gray_frame,
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Compute the average motion magnitude
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        average_motion = np.mean(mag)

        # Append the average motion to the waveform
        self.flow_waveform.append(average_motion)

        # Update the previous frame
        self.previous_frame = gray_frame

        return average_motion

    def get_waveform(self):
        return self.flow_waveform

    def reset(self):
        self.previous_frame = None
        self.flow_waveform = []