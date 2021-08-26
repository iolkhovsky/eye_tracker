import tensorflow as tf

from dataset.tfrecord_utils import decode_fn


class LabelEncoder:

    def __init__(self, prediction_encoder, img_normalizer=None):
        self._encoder = prediction_encoder
        self.normalizer = img_normalizer

    def __str__(self):
        return "PupilMarkupLabelEncoder"

    def encode(self, batch):
        batch = decode_fn(batch)
        img = tf.io.decode_jpeg(batch["image/encoded"])
        img = tf.cast(img, dtype=tf.float32)
        if self.normalizer is not None:
            img = self.normalizer(img)
        label = tf.stack(
            [
                tf.cast(batch["pupil"], dtype=tf.float32),
                batch["canonical/a"],
                batch["canonical/b"],
                batch["canonical/x"],
                batch["canonical/y"],
                batch["canonical/t"]
            ],
            axis=0
        )
        return img, self._encoder(label)

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)
