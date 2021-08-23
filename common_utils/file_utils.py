import yaml
from os.path import dirname, isfile, isdir
from os import makedirs


def read_yaml(path):
    assert isfile(path), f"{path} is not a path to existing file"
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
        return data


def write_yaml(path, data):
    parent_folder = dirname(path)
    if len(parent_folder) and not isdir(parent_folder):
        makedirs(parent_folder)
    with open(path, "w") as f:
        yaml.dump(data, f, Dumper=yaml.SafeDumper)
