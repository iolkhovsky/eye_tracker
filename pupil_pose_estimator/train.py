import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Pupil pose estimator training script")
    parser.add_argument("--config", type=str,
                        help="Absolute path to the *.yml training config file")
    return parser.parse_args()


def run_training(args):
    pass


if __name__ == "__main__":
    run_training(parse_args())
