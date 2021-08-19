import argparse
import cv2
from glob import glob
import numpy as np
from os.path import isdir, join
import sys
from tqdm import tqdm

from utils.file_utils import write_yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Camera calibration tool")
    parser.add_argument("--pattern_photos", "-p", type=str, default="pattern_photos",
                        help="Absolute path to source images with calibration pattern")
    parser.add_argument("--output_file", "-o", type=str, default="camera.yaml",
                        help="Absolute path to the output camera calibration file")
    parser.add_argument("--hor_size", "-hs", type=int, default=9,
                        help="Checkboard pattern horizontal size (corners)")
    parser.add_argument("--ver_size", "-vs", type=int, default=7,
                        help="Checkboard pattern vertical size (corners)")
    return parser.parse_args()


def run_calibration(args):
    assert isdir(args.pattern_photos), f"'--pattern_photos' must be an abs path to the directory with input photos " \
                                       f"of calibration pattern: {args.images}"
    pattern_structure = (args.ver_size, args.hor_size)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints, imgpoints = [], []
    objp = np.zeros((1, pattern_structure[0] * pattern_structure[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:pattern_structure[0], 0:pattern_structure[1]].T.reshape(-1, 2)

    file_paths = glob(join(args.pattern_photos, "*.jpg"))
    with tqdm(total=len(file_paths), file=sys.stdout) as pbar:
        for img_path in file_paths:
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, pattern_structure, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                     cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
            pbar.update(1)

    print("Starting camera calibration...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Calibration completed.")
    data = {
        "camera": mtx.tolist(),
        "dist": dist.tolist(),
        "rvecs": np.asarray(rvecs).tolist(),
        "tvecs": np.asarray(tvecs).tolist(),
    }
    write_yaml(args.output_file, data)
    print(f"Result: \n{data}")


if __name__ == "__main__":
    run_calibration(parse_args())
