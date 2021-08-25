import argparse
from os.path import isdir, isfile
import tensorflow as tf

from common_utils.file_utils import read_yaml
from pupil_pose_estimator.normalization import build_normalizer
from pupil_pose_estimator.metrics import PupilClassificationMetric, PupilePoseEstimationQuality


def parse_args():
    parser = argparse.ArgumentParser(description="Inference model generation script")
    parser.add_argument("--config", type=str,
                        help="Absolute path to the *.yml training config file")
    parser.add_argument("--model", type=str,
                        help="Source model")
    parser.add_argument("--output", type=str, default="inference_model.ckpt",
                        help="Absolute path to the output inference model")
    return parser.parse_args()


def run_model_building(args):
    assert isfile(args.config), f"Config {args.config} doens't exist"
    config = read_yaml(args.config)
    img_size = int(config["model"]["input_size"])
    normalizer = build_normalizer(config)
    assert isdir(args.model), f"Model {args.model} doesn't exist"
    model = tf.keras.models.load_model(args.model, custom_objects={
        "PupilClassificationMetric": PupilClassificationMetric,
        "PupilePoseEstimationQuality": PupilePoseEstimationQuality
    })

    input_tensor = tf.keras.Input(shape=(None, None, 3))
    resized_img = tf.image.resize(input_tensor, img_size)
    normalized_img = normalizer(resized_img)
    prediction = model.predict(normalized_img)

    inference_model = tf.keras.Model(input_tensor, prediction, name="PupilPoseEstimatorInferenceModel")
    inference_model.save(path=args.output)


if __name__ == "__main__":
    run_model_building(parse_args())
