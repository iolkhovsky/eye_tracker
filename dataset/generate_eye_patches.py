import argparse
import cv2
from os import makedirs
from os.path import basename, isdir, isfile, join, splitext
from shutil import rmtree

from utils.haar_detector import HaarEyeDetector, HaarFaceDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Eye patches generator")
    parser.add_argument("--source",
                        help="Absolute path to source videofile",
                        type=str)
    parser.add_argument("--patch_size",
                        help="Generated patches size",
                        default=64,
                        type=int)
    parser.add_argument("--output",
                        help="Absolute path to the output directory",
                        default="eye_patches",
                        type=str)
    parser.add_argument("--patch_format",
                        help="Format of the output image files",
                        default="jpg",
                        type=str)
    return parser.parse_args()


def draw_eyes(in_frame, eyes, color=(0, 255, 0), width=2):
    frame = in_frame.copy()
    for idx, (x, y, w, h) in enumerate(eyes):
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, width)
        cv2.putText(frame, f"Id: {idx}", (x + w + 10, y + h), 0, 0.3, color)
    return frame


def run_eye_patches_generation(args):
    assert isinstance(args.source, str)
    if args.source.isnumeric():
        args.source = int(args.source)
        filename = f"webcamera_{args.source}"
    else:
        assert isfile(args.source), f"Source video file doesn't exist: {args.source}"
        filename = splitext(basename(args.source))[0]
    cap = cv2.VideoCapture(args.source)
    eye_detector = HaarEyeDetector()
    face_detector = HaarFaceDetector()

    if isdir(args.output):
        rmtree(args.output)
    makedirs(args.output)

    patch_idx = 0
    stream_ready = True
    while stream_ready:
        stream_ready, frame = cap.read()
        if not stream_ready:
            break
        visualization = frame.copy()
        for x, y, w, h in face_detector(frame):
            face_subframe = frame[y:y+h, x:x+w, :]
            eyes = eye_detector(face_subframe)
            for ex, ey, ew, eh in eyes:
                patch = face_subframe[ey:ey+eh, ex:ex+ew, :]
                patch = cv2.resize(patch, (args.patch_size, args.patch_size))
                patch_path = join(args.output, f"{filename}_{patch_idx}.{args.patch_format}")
                cv2.imwrite(patch_path, patch)
                print(f"{patch_idx}: Generated patch: {patch_path}")
                patch_idx += 1
            eyes = [(x_eye + x, y_eye + y, w_eye, h_eye) for x_eye, y_eye, w_eye, h_eye in eyes]
            visualization = draw_eyes(frame, eyes, color=(0, 0, 255))
        cv2.imshow("Stream", visualization)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    run_eye_patches_generation(parse_args())
