import numpy as np
import tensorflow as tf


class PredictionCodec:

    def __init__(self, img_size, pupil_anchor=0.3, scale_center=10., scale_size=5.):
        self._img_size = img_size
        self._pupil_anchor = pupil_anchor
        self._scale_center = scale_center
        self._scale_size = scale_size
        self._k_center = self._scale_center / (self._img_size * self._pupil_anchor)
        self._k_size = self._scale_size / (self._img_size * self._pupil_anchor)

    def __str__(self):
        return f"PredictionCodec{self._img_size}"

    def encode(self, target):
        """
        :param target: 2d tensor, row:
        [pupil_conf [0...1], ellipse_a [float], ellipse_b[float], ellipse_x[float], ellipse_y[float], teta[+|-0.5pi]]
        """
        target_shape = target.shape
        assert isinstance(target, tf.Tensor)
        if len(target_shape) < 2:
            target = tf.expand_dims(target, axis=0)
        assert target.shape[1] == 6
        labels = tf.expand_dims(target[:, 0], 1)
        ellipse_axis = target[:, 1:3]
        center_coordinates = target[:, 3:5]
        angles = tf.expand_dims(target[:, 5], 1)
        result = tf.concat(
            [
                labels,
                tf.math.multiply(ellipse_axis, self._k_size),
                tf.math.multiply(center_coordinates, self._k_center),
                tf.math.divide(angles, 0.5 * np.pi)
            ],
            axis=1
        )
        return tf.reshape(result, shape=target_shape)

    def decode(self, prediction):
        """
        :param prediction:
        """
        assert isinstance(prediction, tf.Tensor) and prediction.shape[1] == 6
        labels = tf.expand_dims(prediction[:, 0], 1)
        ellipse_axis = prediction[:, 1:3]
        center_coordinates = prediction[:, 3:5]
        angles = tf.expand_dims(prediction[:, 5], 1)
        return tf.concat(
            [
                labels,
                tf.math.divide(ellipse_axis, self._k_size),
                tf.math.divide(center_coordinates, self._k_center),
                tf.math.multiply(angles, 0.5 * np.pi)
            ],
            axis=1
        )

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)
