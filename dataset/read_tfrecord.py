import argparse
import cv2
import numpy as np
from os import makedirs
from os.path import basename, isdir, isfile, join
from shutil import rmtree
import tensorflow as tf

from common_utils.ellipse import visualize_ellipse
from dataset.tfrecord_utils import decode_fn

def parse_args():
    parser = argparse.ArgumentParser(description="TFRecord validation tool")
    parser.add_argument("--file", type=str,
                        help="Absolute path to the target TFRecord file")
    parser.add_argument("--output", type=str, default="axtracted_data",
                        help="Output directory for extracted data")
    parser.add_argument("--img_size", type=int, default=640,
                        help="Exported images size")
    return parser.parse_args()


def run_extraction(args):
    assert isfile(args.file), f"{args.file} doesn't exist"
    if isdir(args.output):
        rmtree(args.output)
    makedirs(args.output)

    for batch in tf.data.TFRecordDataset([args.file]).map(decode_fn):
        img_buffer = batch["image/encoded"].numpy()
        img = cv2.imdecode(np.frombuffer(img_buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
        filename = basename(batch["image/filename"].numpy()).decode("utf-8")
        visualization = cv2.resize(img, (args.img_size, args.img_size))
        scale = args.img_size / img.shape[0]
        transform = np.asarray([
            [scale, 0, 0.],
            [0, scale, 0.]
        ])
        if batch["pupil"]:
            canonical = (
                batch["canonical/a"].numpy(),
                batch["canonical/b"].numpy(),
                batch["canonical/x"].numpy(),
                batch["canonical/y"].numpy(),
                batch["canonical/t"].numpy(),
            )
            visualization = visualize_ellipse(canonical, visualization, transform)
        cv2.imwrite(join(args.output, filename), visualization)


if __name__ == "__main__":
    run_extraction(parse_args())
