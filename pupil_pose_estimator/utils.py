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


def visualize_pupil(tf_images, tf_labels, denormalizer=None, min_img_size=128):
    out = []
    if denormalizer is not None:
        tf_images = denormalizer(tf_images)
    for img, label in zip(tf_images, tf_labels):
        img = img.numpy().astype(np.uint8)
        img_size = img.shape[0]
        vis_scale = min_img_size / img_size
        transform = None
        if vis_scale > 1.:
            transform = np.asarray([[vis_scale, 0, 0], [0, vis_scale, 0]], dtype=np.float32)
            img = cv2.resize(img, (min_img_size, min_img_size))
        prob = label.numpy()[0]
        canonical = label.numpy()[1:]
        vis_img = visualize_ellipse(canonical, prob, img, transform)
        out.append(tf.convert_to_tensor(vis_img))
    return out
