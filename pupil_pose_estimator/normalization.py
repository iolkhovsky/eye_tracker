import numpy as np
import tensorflow as tf


def normalize_mobilenetv2(img):
    return tf.keras.applications.mobilenet_v2.preprocess_input(img)


def normalize_resnet50(img):
    return tf.keras.applications.resnet50.preprocess_input(img)


def denormalize_mobilenetv2(img):
    img = img + 1.
    img = tf.multiply(img, 127.5)
    return img


def denormalize_resnet50(img):
    img = img + tf.constant(np.array([103.939, 116.779, 123.68]), dtype=tf.float32)
    return img[..., ::-1]


def build_normalizer(config):
    feature_extractor_type = config["model"]["feature_extractor"]
    if feature_extractor_type == "MobileNetV2":
        return normalize_mobilenetv2
    elif feature_extractor_type == "ResNet50":
        return normalize_resnet50
    else:
        raise ValueError(f"{feature_extractor_type} is not available feature extractor")


def build_denormalizer(config):
    feature_extractor_type = config["model"]["feature_extractor"]
    if feature_extractor_type == "MobileNetV2":
        return denormalize_mobilenetv2
    elif feature_extractor_type == "ResNet50":
        return denormalize_resnet50
    else:
        raise ValueError(f"{feature_extractor_type} is not available feature extractor")
