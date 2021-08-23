import argparse
import cv2
import numpy as np
from os import makedirs
from os.path import basename, isdir, isfile, join
from shutil import rmtree
import tensorflow as tf

from common_utils.ellipse import visualize_ellipse


def parse_args():
    parser = argparse.ArgumentParser(description="TFRecord validation tool")
    parser.add_argument("--file", type=str,
                        help="Absolute path to the target TFRecord file")
    parser.add_argument("--output", type=str, default="axtracted_data",
                        help="Output directory for extracted data")
    parser.add_argument("--img_size", type=int, default=640,
                        help="Exported images size")
    return parser.parse_args()


def decode_fn(record_bytes):
    return tf.io.parse_single_example(
        record_bytes,
        features={
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/depth': tf.io.FixedLenFeature([], tf.int64),
            'image/filename': tf.io.FixedLenFeature([], tf.string),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            "pupil": tf.io.FixedLenFeature([], tf.int64),
            "equation/a": tf.io.FixedLenFeature([], tf.float32),
            "equation/b": tf.io.FixedLenFeature([], tf.float32),
            "equation/c": tf.io.FixedLenFeature([], tf.float32),
            "equation/d": tf.io.FixedLenFeature([], tf.float32),
            "equation/e": tf.io.FixedLenFeature([], tf.float32),
            "equation/f": tf.io.FixedLenFeature([], tf.float32),
            "canonical/a": tf.io.FixedLenFeature([], tf.float32),
            "canonical/b": tf.io.FixedLenFeature([], tf.float32),
            "canonical/x": tf.io.FixedLenFeature([], tf.float32),
            "canonical/y": tf.io.FixedLenFeature([], tf.float32),
            "canonical/t": tf.io.FixedLenFeature([], tf.float32),
        }
    )


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
