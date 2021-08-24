import argparse
from os.path import isfile, join
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from common_utils.file_utils import read_yaml
from dataset.label_encoder import LabelEncoder
from pupil_pose_estimator.model import PupilPoseEstimator
from pupil_pose_estimator.loss import PupilEstimatorLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Pupil pose estimator training script")
    parser.add_argument("--config", type=str,
                        help="Absolute path to the *.yml training config file")
    return parser.parse_args()


def build_model(config):
    model_config = config["model"]
    if model_config["pretrained"]["enable"]:
        checkpoint_path = model_config["pretrained"]["path"]
        assert isfile(checkpoint_path), f"{checkpoint_path} doesn't exist"
        return tf.keras.models.load_model(checkpoint_path)
    else:
        return PupilPoseEstimator(
            backbone=model_config["feature_extractor"],
            do_rate=model_config["dropout"]
        )


def run_training(args):
    assert isfile(args.config), f"{args.config} doesn't exist"
    config = read_yaml(args.config)
    print(f"Loaded config: \n{config}")
    model = build_model(config)
    model.build(input_shape=(None, config["model"]["input_size"], config["model"]["input_size"], 3))
    model.summary()

    loss = PupilEstimatorLoss(class_w=config["training"]["loss_weights"]["classification"],
                              regr_w=config["training"]["loss_weights"]["regression"])

    normalizer = None  # TODO build_normalizer()
    encoder = LabelEncoder(normalizer)
    train_dataset = tf.data.TFRecordDataset(config["dataset"]["train"]["tfrecords"])
    train_dataset = train_dataset.map(encoder).batch(config["dataset"]["train"]["batch_size"])
    val_datset = tf.data.TFRecordDataset(config["dataset"]["val"]["tfrecords"])
    val_datset = val_datset.map(encoder).batch(config["dataset"]["val"]["batch_size"])

    epochs = config["training"]["epochs"]

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(join(config["training"]["checkpoints_path"], "ep{epoch}_ckpt")),
        tf.keras.callbacks.TensorBoard(log_dir=config["training"]["logs_path"])
    ]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=loss,
    )
    model.fit(
        train_dataset, epochs=epochs, callbacks=callbacks, validation_data=val_datset,
    )


if __name__ == "__main__":
    run_training(parse_args())
