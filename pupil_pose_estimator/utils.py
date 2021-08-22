import numpy as np
import tensorflow as tf


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
