import argparse
from asyncore import write
import os
import yaml
import numpy as np
import shutil

import util

type_choices = ["flickr"]

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--type', required=True, choices=type_choices)

    args = parser.parse_args()
    return args

def parse_train_test_split(train_test_split):
    splits = train_test_split.split(":")
    train = int(splits[0])
    test = int(splits[1])

    if not (train + test == 100):
        raise RuntimeError("Illegal train test split")

    return train / 100

def validate_classes(classes, num_classes):
    found_labels = set()
    found_classes = set()
    found_subdirs = set()

    if num_classes != len(classes):
        raise RuntimeError("Num classes does not match")

    for c in classes:
        label = c["label"]
        class_name = c["class_name"]
        class_dirs = c["class_dirs"]

        if label in found_labels:
            raise RuntimeError("Duplicate label found: " + str(label))
        
        if class_name in found_classes:
            raise RuntimeError("Duplicate class name found: " + class_name)

        for subdir in class_dirs:
            dirname = subdir.replace("/", "")
            dirname = dirname.replace(".", "")
            if dirname in found_subdirs:
                raise RuntimeError("Duplicate dir found: ", + subdir)

            found_subdirs.add(dirname)
        
        found_labels.add(label)
        found_classes.add(class_name)

    assert len(found_labels) == num_classes
    for i in range(num_classes):
        assert i in found_labels

def get_filenames(input_dir, sub_dirs):
    filenames = []
    for dirname in sub_dirs:
        subdir = os.path.join(input_dir, dirname)

        for filename in os.listdir(subdir):
            if not filename.endswith(".png"):
                continue
            
            filenames.append(os.path.join(subdir, filename))

    return filenames

def create_flickr_data(config_file):
    with open(config_file) as f:
        config = yaml.safe_load(f)

    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    train_test_split = config["train_test_split"]
    classes = config["classes"]
    num_classes = config["num_classes"]
    img_size = config["img_size"]

    if not os.path.exists(input_dir):
        raise RuntimeError("input dir does not exists")

    train_split = parse_train_test_split(train_test_split)
    validate_classes(classes, num_classes)

    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_labels = []
    test_labels = []

    label_map = dict()

    for c in classes:
        label = c["label"]
        class_name = c["class_name"]
        class_dirs = c["class_dirs"]

        label_map[label] = class_name

        class_files = get_filenames(input_dir, class_dirs)

        num_train = int(train_split * len(class_files))
        train_inds = np.random.choice(len(class_files), size=(num_train), replace=False)

        for i in range(len(class_files)):
            if i in train_inds:
                dest_dir = train_dir
                dest_labels = train_labels
            else:
                dest_dir = test_dir
                dest_labels = test_labels

            dest_path = os.path.join(dest_dir, os.path.basename(class_files[i]))
            shutil.copyfile(class_files[i], dest_path)

            dest_labels.append([dest_path, label])

    train_label_file = os.path.join(output_dir, 'train_labels.pkl')
    test_label_file = os.path.join(output_dir, 'test_labels.pkl')

    util.write_file(train_label_file, {'num_classes':num_classes, 'img_size': img_size, 'label_map': label_map, 'labels': train_labels})
    util.write_file(test_label_file, {'num_classes':num_classes, 'img_size': img_size, 'label_map': label_map, 'labels': test_labels})

if __name__ == "__main__":
    args = parse_args()

    config_file = args.config_file
    type = args.type

    if not os.path.exists(config_file):
        raise RuntimeError('Invalid config file')

    if type == "flickr":
        create_flickr_data(config_file)
    else:
        raise RuntimeError("Illegal type found: " + type)

