import tensorflow as tf


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_feature_list(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def decode_fn(record_bytes):
    return tf.io.parse_single_example(
        record_bytes,
        features={
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/depth': tf.io.FixedLenFeature([], tf.int64),
            'image/filename': tf.io.FixedLenFeature([], tf.string),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            "pupil": tf.io.FixedLenFeature([], tf.int64),
            "equation/a": tf.io.FixedLenFeature([], tf.float32),
            "equation/b": tf.io.FixedLenFeature([], tf.float32),
            "equation/c": tf.io.FixedLenFeature([], tf.float32),
            "equation/d": tf.io.FixedLenFeature([], tf.float32),
            "equation/e": tf.io.FixedLenFeature([], tf.float32),
            "equation/f": tf.io.FixedLenFeature([], tf.float32),
            "canonical/a": tf.io.FixedLenFeature([], tf.float32),
            "canonical/b": tf.io.FixedLenFeature([], tf.float32),
            "canonical/x": tf.io.FixedLenFeature([], tf.float32),
            "canonical/y": tf.io.FixedLenFeature([], tf.float32),
            "canonical/t": tf.io.FixedLenFeature([], tf.float32),
        }
    )
