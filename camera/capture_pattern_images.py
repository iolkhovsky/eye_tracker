import argparse
import cv2
from os.path import isfile, isdir, join
from os import makedirs
from shutil import rmtree

import cv2.cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Photo capturing tool")
    parser.add_argument("--source", "-s", type=str, default="0",
                        help="Source of the video stream")
    parser.add_argument("--output", "-o", type=str, default="pattern_photos",
                        help="Path to the output directory")
    parser.add_argument("--max_cnt", "-m", type=int, default=100,
                        help="Max amount of photos being exported")
    parser.add_argument("--frame_period", "-p", type=int, default="500",
                        help="Minimum frame period (ms)")
    parser.add_argument("--hor_size", "-hs", type=int, default=9,
                        help="Checkboard pattern horizontal size (corners)")
    parser.add_argument("--ver_size", "-vs", type=int, default=7,
                        help="Checkboard pattern vertical size (corners)")
    return parser.parse_args()


def run_capture(args):
    assert isinstance(args.source, str)
    if args.source.isnumeric():
        args.source = int(args.source)
    else:
        assert isfile(args.source), f"'--source' must be an abs path to the video file: {args.images}"
    if isdir(args.output):
        rmtree(args.output)
    makedirs(args.output)
    frame_ret, photos_cnt = True, 0
    cap = cv2.VideoCapture(args.source)
    pattern_structure = (args.ver_size, args.hor_size)
    term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    while frame_ret and photos_cnt < args.max_cnt:
        frame_ret, frame = cap.read()
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(grayscale, pattern_structure, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                 cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            path = join(args.output, f"{photos_cnt}.jpg")
            cv2.imwrite(path, frame)
            print(f"{photos_cnt}: {path} has been saved")
            photos_cnt += 1
            corners = cv2.cornerSubPix(grayscale, corners, (11, 11), (-1, -1), term_criteria)
            frame = cv2.drawChessboardCorners(frame, pattern_structure, corners, ret)
        cv2.imshow("Photo", frame)
        if cv2.waitKey(args.frame_period) == ord('q'):
            break


if __name__ == "__main__":
    run_capture(parse_args())
