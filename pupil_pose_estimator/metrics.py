import tensorflow as tf


class PupilClassificationMetric(tf.keras.metrics.Metric):

    def __init__(self):
        super(PupilClassificationMetric, self).__init__(
            name="PupilClassificationMetric"
        )
        self._accuracy = tf.keras.metrics.Accuracy()
        self._result = 0.

    def update_state(self, y_true, y_pred, sample_weight=None):
        labels = y_true[:, 0]
        preds = y_pred[:, 0]
        self._accuracy.reset_state()
        self._accuracy.update_state(labels, preds)
        self._result = self._accuracy.result()

    def result(self):
        return self._result


class PupilePoseEstimationQuality(tf.keras.metrics.Metric):

    def __init__(self):
        super(PupilePoseEstimationQuality, self).__init__(
            name="PupilePoseEstimationQuality"
        )
        self._mse = tf.keras.losses.MeanSquaredError()
        self._result = 0.

    def update_state(self, y_true, y_pred, sample_weight=None):
        target_values = y_true[:, 1:]
        pred_values = y_pred[:, 1:]
        self._result = self._mse(target_values, pred_values)

    def result(self):
        return -1. * self._result
