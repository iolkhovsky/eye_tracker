import tensorflow as tf


DEFAULT_FEXT = "MobileNetV2"


class PupilClassificationHead(tf.keras.layers.Layer):

    def __init__(self, *args, **kwargs):
        super(PupilClassificationHead, self).__init__(name="PupilClassificationHead", *args, **kwargs)
        self.dense = tf.keras.layers.Dense(units=1)

    def __call__(self, features, training=False):
        logits = self.dense(features)
        return logits


class PupilRegressionHead(tf.keras.layers.Layer):

    def __init__(self, *args, **kwargs):
        super(PupilRegressionHead, self).__init__(name="PupilRegressionHead", *args, **kwargs)
        self.dense = tf.keras.layers.Dense(units=6)
        self.act = tf.keras.layers.Activation(tf.keras.activations.sigmoid)

    def __call__(self, features, training=False):
        x = self.dense(features)
        return self.act(x)


class PupilRegressor(tf.keras.Model):

    def __init__(self, backbone=DEFAULT_FEXT, **kwargs):
        super(PupilRegressor, self).__init__(name="PupilRegressor", **kwargs)
        self.backbone = getattr(tf.keras.applications, backbone)(include_top=False, input_shape=[None, None, 3],
                                                                 weights="imagenet")
        self.flatten = tf.keras.layers.Flatten()
        self.bn = tf.keras.layers.BatchNormalization()
        self.classification_head = PupilClassificationHead()
        self.regression_head = PupilRegressionHead()

    def call(self, image, training=False):
        features = self.backbone(image, training=training)
        features = self.flatten(features)
        features = self.bn(features)
        logit = self.classification_head(features, training=training)
        coeffs = self.regression_head(features, training=training)
        return tf.concat(
            (logit, coeffs),
            axis=-1
        )
