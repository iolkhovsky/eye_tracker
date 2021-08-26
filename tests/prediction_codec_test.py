import numpy as np
import pytest
import tensorflow as tf

from pupil_pose_estimator.prediction_codec import PredictionCodec


def test_prediction_coding():
    img_size = 64
    pupil_anchor = 0.3
    scale_center = 10.
    scale_size = 5.
    values = tf.convert_to_tensor(
        [
            [1., 32, 16, 32, 32, 0.3],
            [1., 4., 2., 10, 40, -0.1]
        ],
        dtype=tf.float32
    )
    codec = PredictionCodec(img_size=img_size,
                            pupil_anchor=pupil_anchor,
                            scale_center=scale_center,
                            scale_size=scale_size)

    encoded = codec.encode(values)
    for encoded_sample, sample in zip(encoded, values):
        enc_label, enc_a, enc_b, enc_x, enc_y, enc_teta = encoded_sample.numpy()
        label, a, b, x, y, teta = sample.numpy()
        assert label == enc_label
        assert a * scale_size / (pupil_anchor * img_size) == pytest.approx(enc_a, 1e-2)
        assert b * scale_size / (pupil_anchor * img_size) == pytest.approx(enc_b, 1e-2)
        assert x * scale_center / (pupil_anchor * img_size) == pytest.approx(enc_x, 1e-2)
        assert y * scale_center / (pupil_anchor * img_size) == pytest.approx(enc_y, 1e-2)
        assert teta / (0.5 * np.pi) == pytest.approx(enc_teta, 1e-2)


def test_prediction_decoding():
    values = tf.convert_to_tensor(
        [
            [1., 32, 16, 32, 32, 0.3],
            [1., 4., 2., 10, 40, -0.1]
        ],
        dtype=tf.float32
    )
    codec = PredictionCodec(img_size=64)
    encoded = codec.encode(values)
    decoded = codec.decode(encoded)
    tf.debugging.assert_near(values, decoded, rtol=1e-2)
