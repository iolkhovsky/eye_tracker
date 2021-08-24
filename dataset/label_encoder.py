import cv2
import numpy as np
import tensorflow as tf

from dataset.tfrecord_utils import decode_fn


class LabelEncoder:

    def __init__(self, img_normalizer=None):
        self.normalizer = img_normalizer

    def __str__(self):
        return "PupilMarkupLabelEncoder"

    def encode(self, batch):
        batch = decode_fn(batch)
        img = tf.io.decode_jpeg(batch["image/encoded"])
        img = tf.cast(img, dtype=tf.float32)
        img_size = tf.cast(batch["image/width"], tf.float32)
        if self.normalizer is not None:
            img = self.normalizer(img)
        label = tf.stack(
            [
                tf.cast(batch["pupil"], dtype=tf.float32),
                batch["canonical/a"] / img_size,
                batch["canonical/b"] / img_size,
                batch["canonical/x"] / img_size,
                batch["canonical/y"] / img_size,
                batch["canonical/t"] / tf.constant(0.5 * np.pi, dtype=tf.float32)
            ],
            axis=0
        )
        return img, label
