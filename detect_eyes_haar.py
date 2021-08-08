import argparse
import cv2
import numpy as np

from utils.haar_detector import HaarEyeDetector, HaarFaceDetector


WEBCAM_SOURCE = "webcam"
VIDEOFILE_SOURCE = "videofile"


def parse_args():
    parser = argparse.ArgumentParser(description="Haar eye detector demo script")
    parser.add_argument("--source",
                        help="Source of the video ('videofile' or 'webcamera')",
                        default=f"{WEBCAM_SOURCE}",
                        type=str)
    parser.add_argument("--id",
                        help="Id of the videosource (path for videofile, system id for web-camera)",
                        default=0)
    parser.add_argument("--eye_det_config",
                        help="Absolute path to cascade classifier's config (*.xml)",
                        default=None)
    parser.add_argument("--face_det_config",
                        help="Absolute path to cascade classifier's config (*.xml)",
                        default=None)
    return parser.parse_args()


def draw_eyes(in_frame, eyes, color=(0, 255, 0), width=2):
    assert isinstance(in_frame, np.ndarray)
    assert len(in_frame.shape) == 3
    frame = in_frame.copy()
    for idx, (x, y, w, h) in enumerate(eyes):
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, width)
        cv2.putText(frame, f"Id: {idx}", (x + w + 10, y + h), 0, 0.3, color)
    return frame


def run_haar_eye_detector(args):
    source = None
    if args.source == WEBCAM_SOURCE:
        source = cv2.VideoCapture(int(args.id))
    elif args.source == VIDEOFILE_SOURCE:
        source = cv2.VideoCapture(str(args.id))
    else:
        raise f"Invalid video source: {source}"
    config = None
    eye_detector = HaarEyeDetector(args.eye_det_config)
    face_detector = HaarFaceDetector(args.face_det_config)
    stream_ready = True
    while stream_ready:
        stream_ready, frame = source.read()
        visualization = frame.copy()
        for x, y, w, h in face_detector(frame):
            face_subframe = frame[y:y+h, x:x+w, :]
            eyes = eye_detector(face_subframe)
            eyes = [(x_eye + x, y_eye + y, w_eye, h_eye) for x_eye, y_eye, w_eye, h_eye in eyes]
            visualization = draw_eyes(frame, eyes, color=(0, 0, 255))
        cv2.imshow("Stream", visualization)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break


if __name__ == "__main__":
    run_haar_eye_detector(parse_args())
