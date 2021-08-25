import pytest
import tensorflow as tf

from pupil_pose_estimator.model import PupilPoseEstimator


def test_pose_estimator_model():
    model = PupilPoseEstimator(
        backbone="MobileNetV2",
        do_rate=0.5,
    )
    batch_size = 2
    test_input = tf.random.uniform(shape=(batch_size, 64, 64, 3), dtype=tf.float32)

    preds_logit = model(test_input)
    preds = model.predict(test_input)

    assert preds_logit.shape[0] == batch_size
    assert preds_logit.shape[1] == 6

    for logit, a, b, x, y, teta in preds_logit:
        assert 0. <= a <= 1.
        assert 0. <= b <= 1.
        assert 0. <= x <= 1.
        assert 0. <= y <= 1.
        assert -1. <= teta <= 1.

    for pred, pred_logit in zip(preds, preds_logit):
        assert tf.reduce_all(pred[1:] == pred_logit[1:])
        assert tf.nn.sigmoid(pred_logit[0]) == pred[0]
