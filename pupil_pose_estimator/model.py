import tensorflow as tf


DEFAULT_FEXT = "MobileNetV2"


class PupilPoseEstimator(tf.keras.Model):

    def __init__(self, backbone=DEFAULT_FEXT, **kwargs):
        super(PupilPoseEstimator, self).__init__(name="PupilPoseEstimator", **kwargs)
        self.backbone = getattr(tf.keras.applications, backbone)(include_top=False, input_shape=[None, None, 3],
                                                                 weights="imagenet")
        self.flatten = tf.keras.layers.Flatten()
        self.bn = tf.keras.layers.BatchNormalization()
        self.dense = tf.keras.layers.Dense(units=6)
        self.clf_act = tf.keras.layers.Activation(tf.nn.sigmoid)
        self.angle_act = tf.keras.layers.Activation(tf.nn.tanh)
        self.spatial_act = tf.keras.layers.Activation(tf.math.exp)

    def call(self, image, training=False):
        features = self.backbone(image, training=training)
        features = self.flatten(features)
        features = self.bn(features)
        return self.dense(features)

    def predict(self, image):
        prediction = self.call(image, training=False)
        pupile_conf = self.clf_act(prediction[:, 0])
        spatial_pars = self.spatial_act(prediction[:, 1:5])
        angle = self.angle_act(prediction[:, 5])
        return tf.concat(
            (tf.expand_dims(pupile_conf, axis=1), spatial_pars, tf.expand_dims(angle, axis=1)),
            axis=1
        )
