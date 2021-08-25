import numpy as np
import pytest
import tensorflow as tf

from pupil_pose_estimator.loss import PupilEstimatorLoss


def test_pose_estimator_loss():
    class_w, regr_w = 1., 1.
    loss = PupilEstimatorLoss(
        class_w=class_w,
        regr_w=regr_w,
        from_logits=True
    )
    img_sz = 32.
    target_label = 1.
    target_a = 2.
    target_b = 1.
    target_x = 10.
    target_y = 20.
    target_teta = 30 * np.pi / 180.

    target = tf.convert_to_tensor(
        [[target_label,
          target_a / img_sz,
          target_b / img_sz,
          target_x / img_sz,
          target_y / img_sz,
          target_teta / (0.5 * np.pi)]]
    )

    prediction_logit = tf.convert_to_tensor(
        [[1.15, 2.5 / img_sz, 0.5 / img_sz, 15 / img_sz, 7 / img_sz, -1./6.]]
    )
    loss_value = loss(target, prediction_logit)

    def cross_entropy(target, logit):
        value = tf.nn.sigmoid(logit)
        if target:
            return -1. * tf.math.log(value)
        else:
            return -1. * tf.math.log(1. - value)

    target_clf_loss = cross_entropy(target[0, 0], prediction_logit[0, 0]) * class_w
    target_regr_loss = sum([(t - p) ** 2 for t, p in zip(target[0][1:], prediction_logit[0][1:])]) * class_w / len(target[0][1:])

    target_loss = target_clf_loss + target_regr_loss

    assert loss_value.numpy()[0] == pytest.approx(target_loss.numpy(), 1e-2)
