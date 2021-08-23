import argparse
from glob import glob
from os.path import isdir, join, splitext
from random import shuffle
import tensorflow as tf
from tqdm import tqdm

from dataset.tfrecord_utils import float_feature, int64_feature, bytes_feature
from common_utils.file_utils import read_yaml


def create_tf_example(annotation):
    img_path = annotation["source_image_path"]
    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    feature = {
        'image/height': int64_feature(annotation["image_height"]),
        'image/width': int64_feature(annotation["image_width"]),
        'image/depth': int64_feature(annotation["image_channels"]),
        'image/filename': bytes_feature(img_path.encode('utf8')),
        'image/encoded': bytes_feature(encoded_jpg),
        "pupil": int64_feature(annotation["markup"]["pupil"]),
        "equation/a": float_feature(annotation["markup"]["equation"]["a"]),
        "equation/b": float_feature(annotation["markup"]["equation"]["b"]),
        "equation/c": float_feature(annotation["markup"]["equation"]["c"]),
        "equation/d": float_feature(annotation["markup"]["equation"]["e"]),
        "equation/e": float_feature(annotation["markup"]["equation"]["d"]),
        "equation/f": float_feature(annotation["markup"]["equation"]["f"]),
        "canonical/a": float_feature(annotation["markup"]["canonical"]["a"]),
        "canonical/b": float_feature(annotation["markup"]["canonical"]["b"]),
        "canonical/x": float_feature(annotation["markup"]["canonical"]["x"]),
        "canonical/y": float_feature(annotation["markup"]["canonical"]["y"]),
        "canonical/t": float_feature(annotation["markup"]["canonical"]["teta"]),
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
    train_writer = tf.io.TFRecordWriter(train_dataset_path)
    val_writer = tf.io.TFRecordWriter(val_dataset_path) if val_dataset_path is not None else None
    test_writer = tf.io.TFRecordWriter(test_dataset_path) if test_dataset_path is not None else None

    imgs_list = glob(join(args.dataset, "*.jpg"))
    annotations_list = glob(join(args.dataset, "*.yml"))
    samples_list = list(
        set([splitext(x)[0] for x in imgs_list]).intersection(
        set([splitext(x)[0] for x in annotations_list]))
    )
    total_samples = len(samples_list)
    print(f"Found {total_samples} samples at: {args.dataset}")
    shuffle(samples_list)
    val_samples_cnt = int(args.val_share * total_samples) if args.val_share is not None else 0
    test_samples_cnt = int(args.test_share * total_samples) if args.test_share is not None else 0
    train_samples_cnt = total_samples - val_samples_cnt - test_samples_cnt

    train_samples = samples_list[:train_samples_cnt]
    val_samples = samples_list[train_samples_cnt:train_samples_cnt + val_samples_cnt]
    test_samples = samples_list[train_samples_cnt + val_samples_cnt:]

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
