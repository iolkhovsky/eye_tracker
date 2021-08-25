import cv2
import numpy as np
import tensorflow as tf

from common_utils.ellipse import visualize_ellipse


def decode_estimator_prediction(prediction):
    if isinstance(prediction, tf.Tensor):
        prediction = prediction.numpy()
    return [{
        "conf": p[0],
        "ellipse": {
            "a": p[1],
            "b": p[2],
            "x": p[3],
            "y": p[4],
            "teta": 0.5 * np.pi * p[5]
        }
    } for p in prediction]


def visualize_pupil(tf_images, tf_labels, denormalizer=None):
    out = []
    if denormalizer is not None:
        tf_images = denormalizer(tf_images)
    for img, label in zip(tf_images, tf_labels):
        img = img.numpy().astype(np.uint8)
        img_size = img.shape[0]
        prob, a, b, x, y, teta = label.numpy()
        canonical = a * img_size, b * img_size, x * img_size, y * img_size, teta * np.pi * 0.5
        vis_img = visualize_ellipse(canonical, img)
        out.append(tf.convert_to_tensor(vis_img))
    return out
