import argparse
from glob import glob
from os.path import isdir, join, splitext
from random import shuffle
import tensorflow as tf
from tqdm import tqdm

from utils.file_utils import read_yaml


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _float_feature_list(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_tf_example(annotation):
    img_path = annotation["source_image_path"]
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    img_width = annotation["image_width"]
    img_height = annotation["image_height"]
    img_channels = annotation["image_channels"]
    feature = {
        'image/height': _int64_feature(img_height),
        'image/width': _int64_feature(img_width),
        'image/depth': _int64_feature(img_channels),
        'image/filename': _bytes_feature(img_path.encode('utf8')),
        'image/encoded': _bytes_feature(encoded_jpg),
        "equation/a": _float_feature(annotation["markup"]["equation"]["a"]),
        "equation/b": _float_feature(annotation["markup"]["equation"]["b"]),
        "equation/c": _float_feature(annotation["markup"]["equation"]["c"]),
        "equation/d": _float_feature(annotation["markup"]["equation"]["e"]),
        "equation/e": _float_feature(annotation["markup"]["equation"]["d"]),
        "equation/f": _float_feature(annotation["markup"]["equation"]["f"]),
        "canonical/a": _float_feature(annotation["markup"]["canonical"]["a"]),
        "canonical/b": _float_feature(annotation["markup"]["canonical"]["b"]),
        "canonical/x": _float_feature(annotation["markup"]["canonical"]["x"]),
        "canonical/y": _float_feature(annotation["markup"]["canonical"]["y"]),
        "canonical/t": _float_feature(annotation["markup"]["canonical"]["t"]),

    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    return tf_example


def parse_args():
    parser = argparse.ArgumentParser(description="Pupil markup TFRecord dataset generation")
    parser.add_argument("--dataset", type=str,
                        help="Absolute path to the directory with source images/annotations")
    parser.add_argument("--output", type=str, default="dataset.tfrecord",
                        help="Generated patches size")
    parser.add_argument("--val_share", type=float, default=0.1)
    parser.add_argument("--test_share", type=float, default=0.1)
    return parser.parse_args()


def run_tfrecord_generation(args):
    assert isdir(args.dataset), f"{args.dataset} folder doesn't exist"
    train_dataset_path, val_dataset_path, test_dataset_path = args.output, None, None
    if args.val_share:
        val_dataset_path = splitext(train_dataset_path)[0] + "_val.tfrecord"
    if args.val_share:
        test_dataset_path = splitext(train_dataset_path)[0] + "_test.tfrecord"
    if args.val_share + args.test_share > 0:
        train_dataset_path = splitext(train_dataset_path)[0] + "_train.tfrecord"
    train_writer = tf.python_io.TFRecordWriter(train_dataset_path)
    val_writer = tf.python_io.TFRecordWriter(val_dataset_path) if val_dataset_path is not None else None
    test_writer = tf.python_io.TFRecordWriter(test_dataset_path) if test_dataset_path is not None else None

    imgs_list = glob(join(args.dataset, "*.jpg"))
    annotations_list = glob(join(args.dataset, "*.yml"))
    samples_list = set([splitext(x)[0] for x in imgs_list]).intersection(
        set([splitext(x)[0] for x in annotations_list]))
    total_samples = len(samples_list)
    print(f"Found {total_samples} samples at: {args.dataset}")
    shuffle(samples_list)
    val_samples_cnt = int(args.val_share) * total_samples if args.val_share is not None else 0
    test_samples_cnt = int(args.test_share) * total_samples if args.val_share is not None else 0
    train_samples_cnt = total_samples - val_samples_cnt - test_samples_cnt

    train_samples = samples_list[:train_samples_cnt]
    val_samples = samples_list[train_samples_cnt:train_samples + val_samples_cnt]
    test_samples = samples_list[train_samples + val_samples_cnt:]

    for samples, writer, hint in [(train_samples, train_writer, "train"),
                                  (val_samples, val_writer, "val"),
                                  (test_samples, test_writer, "test")]:
        if writer is None or len(samples) == 0:
            return
        print(f"Generating {hint} dataset - {len(samples)} samples")
        with tqdm(desc=f"{hint} dataset generation", total=len(samples)) as pbar:
            for sample in samples:
                tf_example = create_tf_example(read_yaml(sample + ".yml"))
                writer.write(tf_example.SerializeToString())
                pbar.update(1)


if __name__ == '__main__':
    run_tfrecord_generation(parse_args())
