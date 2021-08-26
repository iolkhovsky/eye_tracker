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

    for batch in tf.data.TFRecordDataset([args.file]).map(decode_fn).batch(2):
        img_batch = batch["image/encoded"]
        filename_batch = batch["image/filename"]
        pupil_batch = batch["pupil"]
        a_batch = batch["canonical/a"]
        b_batch = batch["canonical/b"]
        x_batch = batch["canonical/x"]
        y_batch = batch["canonical/y"]
        teta_batch = batch["canonical/t"]
        for img, filename, a, b, x, y, teta, pupil in zip(img_batch, filename_batch, a_batch, b_batch, x_batch, y_batch, teta_batch, pupil_batch):
            img_buffer = img.numpy()
            img = cv2.imdecode(np.frombuffer(img_buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
            filename = basename(filename.numpy()).decode("utf-8")
            visualization = cv2.resize(img, (args.img_size, args.img_size))
            scale = args.img_size / img.shape[0]
            transform = np.asarray([
                [scale, 0, 0.],
                [0, scale, 0.]
            ])
            if pupil.numpy():
                canonical = (
                    a.numpy(),
                    b.numpy(),
                    x.numpy(),
                    y.numpy(),
                    teta.numpy(),
                )
                visualization = visualize_ellipse(canonical, pupil.numpy(), visualization, transform)
            cv2.imwrite(join(args.output, filename), visualization)


if __name__ == "__main__":
    run_extraction(parse_args())
