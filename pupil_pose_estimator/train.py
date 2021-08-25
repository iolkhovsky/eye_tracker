import argparse
import datetime
from os.path import isfile, join
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from common_utils.file_utils import read_yaml
from dataset.label_encoder import LabelEncoder
from pupil_pose_estimator.loss import PupilEstimatorLoss
from pupil_pose_estimator.model import PupilPoseEstimator
from pupil_pose_estimator.normalization import build_normalizer, build_denormalizer
from pupil_pose_estimator.metrics import PupilClassificationMetric, PupilePoseEstimationQuality
from pupil_pose_estimator.utils import visualize_pupil


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

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    logs_path = join(config["training"]["logs_path"], timestamp)
    ckpt_path = join(config["training"]["checkpoints_path"], timestamp)

    normalizer = build_normalizer(config)
    denormalizer = build_denormalizer(config)
    encoder = LabelEncoder(normalizer)
    train_dataset = tf.data.TFRecordDataset(config["dataset"]["train"]["tfrecords"])
    train_dataset = train_dataset.map(encoder).batch(config["dataset"]["train"]["batch_size"])
    val_dataset = tf.data.TFRecordDataset(config["dataset"]["val"]["tfrecords"])
    val_dataset = val_dataset.map(encoder).batch(config["dataset"]["val"]["batch_size"])

    epochs = config["training"]["epochs"]

    def visualize_prediction(epoch, logs):
        batch = next(iter(val_dataset))
        val_imgs, val_labels = batch
        val_preds = model.predict(val_imgs)
        target_visualization = visualize_pupil(val_imgs, val_labels, denormalizer)
        pred_visualization = visualize_pupil(val_imgs, val_preds, denormalizer)
        writer = tf.summary.create_file_writer(join(logs_path, "visualization"))
        with writer.as_default():
            tf.summary.image("target", target_visualization, step=epoch)
            tf.summary.image("prediction", pred_visualization, step=epoch)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(join(ckpt_path, "ep{epoch}_ckpt")),
        tf.keras.callbacks.TensorBoard(log_dir=logs_path),
        tf.keras.callbacks.LambdaCallback(on_epoch_end=visualize_prediction)
    ]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(float(config["training"]["lr"])),
        loss=PupilEstimatorLoss(
            class_w=config["training"]["loss_weights"]["classification"],
            regr_w=config["training"]["loss_weights"]["regression"],
            from_logits=True
        ),
        metrics=[PupilClassificationMetric(), PupilePoseEstimationQuality()]
    )
    model.fit(
        train_dataset, epochs=epochs, callbacks=callbacks, validation_data=val_dataset,
    )


if __name__ == "__main__":
    run_training(parse_args())
