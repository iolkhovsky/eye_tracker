import argparse
from os.path import isfile
import tensorflow as tf

from pupil_pose_estimator.model import PupilPoseEstimator
from common_utils.file_utils import read_yaml


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


if __name__ == "__main__":
    run_training(parse_args())
