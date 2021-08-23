from abc import ABC, abstractmethod
import cv2
import numpy as np
from os.path import isfile, join
from pathlib import Path


FACE_DETECTOR = "face"
EYE_DETECTOR = "eye"


class HaarDetector(ABC):

    def __init__(self, config_path):
        assert isfile(config_path), f"Cascade classifier's configuration file <{config_path}> doesn't exist"
        self.config_path = config_path
        self.cascade_classifier = cv2.CascadeClassifier()
        self.cascade_classifier.load(config_path)

    def __str__(self):
        return f"HaarDetector<{self.config_path}>"

    def __call__(self, *args, **kwargs):
        return self.detect(*args, **kwargs)

    def detect(self, in_frame):
        assert isinstance(in_frame, np.ndarray), f"Frame should have np.ndarray type. Got {type(in_frame)} instead"
        assert len(in_frame.shape) <= 3, f"Frame should have 2 or 3 axis. Got {len(in_frame.shape)} instead"
        frame = in_frame.copy()
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.cascade_classifier.detectMultiScale(frame)
        return detections


class HaarEyeDetector(HaarDetector):

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = join(Path(__file__).parent.parent.absolute(), r"config/haarcascade_eye.xml")
        super(HaarEyeDetector, self).__init__(config_path=config_path)


class HaarFaceDetector(HaarDetector):

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = join(Path(__file__).parent.parent.absolute(), r"config/haarcascade_frontalface_default.xml")
        super(HaarFaceDetector, self).__init__(config_path=config_path)


def build_haar_detector(detector_type):
    if detector_type == FACE_DETECTOR:
        return HaarFaceDetector
    elif detector_type == EYE_DETECTOR:
        return HaarEyeDetector
    else:
        raise f"Invalid Haar detector type: {detector_type}"
