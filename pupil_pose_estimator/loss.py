import tensorflow as tf


class PupilEstimatorLoss(tf.keras.losses.Loss):

    def __init__(self, class_w=1., regr_w=1., from_logits=True):
        super(PupilEstimatorLoss, self).__init__(
            reduction="none", name="PupilEstimatorLoss"
        )
        self.class_w = class_w
        self.regr_w = regr_w
        self.class_loss = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)
        self.regr_loss = tf.keras.losses.MeanSquaredError()

    def compute(self, y_true, y_pred, sample_weight=None):
        pupil_true, ellipse_true = y_true[:, 0], y_true[:, 1:]
        pupil_pred, ellipse_pred = y_pred[:, 0], y_pred[:, 1:]
        clf_loss = self.class_loss(pupil_true, pupil_pred) * self.class_w
        regr_loss = self.regr_loss(ellipse_true, ellipse_pred) * self.regr_w
        return clf_loss + regr_loss * pupil_true

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)
