import tensorflow as tf


DEFAULT_FEXT = "MobileNetV2"


class PupilPoseEstimator(tf.keras.Model):

    def __init__(self, backbone=DEFAULT_FEXT, do_rate=0.5, **kwargs):
        super(PupilPoseEstimator, self).__init__(name="PupilPoseEstimator", **kwargs)
        self.backbone = getattr(tf.keras.applications, backbone)(include_top=False, input_shape=[None, None, 3],
                                                                 weights="imagenet")
        self.flatten = tf.keras.layers.Flatten()
        self.do = tf.keras.layers.Dropout(rate=do_rate)
        self.bn = tf.keras.layers.BatchNormalization()
        self.dense = tf.keras.layers.Dense(units=6)
        self.angle_act = tf.keras.layers.Activation(tf.nn.tanh)

    def call(self, image, training=False):
        features = self.backbone(image, training=training)
        features = self.flatten(features)
        features = self.bn(features)
        features = self.do(features)
        features = self.dense(features)
        return tf.concat(
            (
                features[:, :5],
                tf.expand_dims(self.angle_act(features[:, 5]), axis=1)
            ),
            axis=1
        )
