import argparse
from asyncore import write
import os
import numpy as np
import shutil
from PIL import Image
import torch
import torchvision.transforms as transforms

import util
from options import TestOptions
import models
from data.base_dataset import get_transform

method_choices = ["build_flickr", "extract_latent"]

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--method', required=True, choices=method_choices)

    args = parser.parse_args()
    return args

def load_image(opt, path):
    path = os.path.expanduser(path)
    img = Image.open(path).convert('RGB')
    transform = get_transform(opt)
    tensor = transform(img).unsqueeze(0)
    return tensor

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

    label_map = dict()

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

        label_map[label] = class_name

    assert len(found_labels) == num_classes
    for i in range(num_classes):
        assert i in found_labels

    return label_map

def parse_and_validate_gan_details(config, label_map, for_extract):
    gan_augment = config['gan_augment']

    if not gan_augment:
        return gan_augment, {}
    
    if config.get("gan_details") is None:
        raise RuntimeError('gan_details required in config when gan_augment is True')

    gan_details = config["gan_details"]

    for class_name in gan_details['class_augment_details']:
        if class_name not in label_map.values():
            raise RuntimeError('Illegal class_name found in gan_details: ' + class_name)

        textures = gan_details['class_augment_details'][class_name]["textures"]
        #TODO fix this, just checking key is there
        num_augment = gan_details['class_augment_details'][class_name]["num_augment"]

        if not for_extract:
            for texture in textures:
                latent_path = os.path.join(gan_details['extracted_latent_code_dir'], texture, 'latent_codes.pth')
                if not os.path.exists(latent_path):
                    raise RuntimeError('Latent code path does not exists: ' + latent_path)

    return gan_augment, gan_details

def get_filenames(input_dir, sub_dirs):
    filenames = []
    for dirname in sub_dirs:
        subdir = os.path.join(input_dir, dirname)

        for filename in os.listdir(subdir):
            if not filename.endswith(".png"):
                continue
            
            filenames.append(os.path.join(subdir, filename))

    return filenames

def get_opt_and_model():
    opt = TestOptions().parse(name_req=False)
    #opt.preprocess = "scale_shortside"
    opt.preprocess = "resize"
    opt.load_size = 512
    opt.crop_size = 512
    opt.name = 'mountain_pretrained'
    opt.dataset_mode = "imagefolder"
    opt.lambda_patch_R1=10.0
    #just for debug
    # opt.evaluation_metrics="texture_extract"
    # opt.result_dir='./results/'
    # opt.texture_mix_alphas=[1.0]
    # opt.method='save_all'
    # opt.latent_mix_alphas=[1.0]
    # opt.latent_type=None
    # opt.input_dir='/home/frc-ag-3/harry_ws/visual_synthesis/final_project/data/flickr/latent_textures'
    # opt.input_structure_image=None
    model = models.create_model(opt)

    return opt, model

def extract_latent_codes(config_file):
    config = util.read_yaml(config_file)
    classes = config["classes"]
    num_classes = config["num_classes"]

    label_map = validate_classes(classes, num_classes)
    gan_augment, gan_details = parse_and_validate_gan_details(config, label_map, True)
    
    if not gan_augment:
        print('do not need to extract latent_codes if gan_augment is False')
        return

    latent_input_dir = gan_details['latent_input_dir']
    extracted_latent_code_dir = gan_details['extracted_latent_code_dir']

    if not os.path.exists(latent_input_dir):
        raise RuntimeError('Invalid latent input dir')

    opt, model = get_opt_and_model()

    for dirname in os.listdir(latent_input_dir):
        subdir = os.path.join(latent_input_dir, dirname)
        if not os.path.isdir(subdir):
            continue

        latent_type = dirname
        output_dir = os.path.join(extracted_latent_code_dir, latent_type)
        os.makedirs(output_dir, exist_ok=True)

        latent_codes = []
        for filename in os.listdir(subdir):
            if not filename.endswith('.png'):
                continue

            image_path = os.path.join(subdir, filename)

            print('Processing: ' + image_path)
            image = load_image(opt, image_path)
        
            #TODO do I need to do this once or every time?
            # Actually do I need this at all?
            model(sample_image=image, command="fix_noise")

            _, texture_code = model(image, command="encode")

            latent_codes.append(texture_code)

        latent_codes = torch.cat(latent_codes, dim=0)
        
        torch.save(latent_codes, os.path.join(output_dir, 'latent_codes.pth'))


def create_flickr_data(config_file):
    config = util.read_yaml(config_file)

    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    train_test_split = config["train_test_split"]
    classes = config["classes"]
    num_classes = config["num_classes"]
    img_size = config["img_size"]

    if not os.path.exists(input_dir):
        raise RuntimeError("input dir does not exists")

    train_split = parse_train_test_split(train_test_split)
    label_map = validate_classes(classes, num_classes)
    gan_augment, gan_details = parse_and_validate_gan_details(config, label_map, False)

    if gan_augment:
        opt, model = get_opt_and_model()

    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_labels = []
    test_labels = []

    train_count = 0
    augment_count = 0

    for c in classes:
        label = c["label"]
        class_name = c["class_name"]
        class_dirs = c["class_dirs"]

        class_files = get_filenames(input_dir, class_dirs)

        num_train = int(train_split * len(class_files))
        train_inds = np.random.choice(len(class_files), size=(num_train), replace=False)

        for i in range(len(class_files)):
            if i in train_inds:
                dest_dir = train_dir
                dest_labels = train_labels
                train_count += 1
            else:
                dest_dir = test_dir
                dest_labels = test_labels

            dest_path = os.path.join(dest_dir, os.path.basename(class_files[i]))
            shutil.copyfile(class_files[i], dest_path)

            dest_labels.append([dest_path, label])

            #only augment training
            if (i in train_inds) and (gan_augment) and (gan_details['class_augment_details'].get(class_name) is not None):
                structure_image = load_image(opt, dest_path)
                structure_code, structure_texture = model(structure_image, command="encode")

                textures = gan_details['class_augment_details'][class_name]["textures"]
                num_augment = gan_details['class_augment_details'][class_name]["num_augment"]

                for texture in textures:
                    latent_path = os.path.join(gan_details['extracted_latent_code_dir'], texture, 'latent_codes.pth')
                    texture_codes = torch.load(latent_path)

                    rand_inds = np.random.choice(texture_codes.shape[0], size=(num_augment), replace=False)

                    for augment_num in range(num_augment):
                        texture_code = texture_codes[rand_inds[augment_num]:rand_inds[augment_num]+1]

                        texture_code = util.lerp(structure_texture, texture_code, gan_details['texture_alpha'])

                        print('Augmenting ' + os.path.basename(dest_path) + ' num ' + str(augment_num))
                        augmented_image = model(structure_code, texture_code, command="decode")
                        augmented_image = transforms.ToPILImage()((augmented_image[0].clamp(-1.0, 1.0) + 1.0) * 0.5)

                        augment_dest_path = os.path.join(dest_dir, os.path.basename(dest_path).split('.png')[0] + '_' + texture + '_' + str(augment_num) + '.png')
                        augmented_image.save(augment_dest_path)

                        assert dest_labels != test_labels
                        dest_labels.append([augment_dest_path, label])
                        augment_count += 1

    assert train_count + augment_count == len(train_labels)

    train_label_file = os.path.join(output_dir, 'train_labels.pkl')
    test_label_file = os.path.join(output_dir, 'test_labels.pkl')

    util.write_file(train_label_file, {'num_classes':num_classes, 'img_size': img_size, 'label_map': label_map, 'labels': train_labels})
    util.write_file(test_label_file, {'num_classes':num_classes, 'img_size': img_size, 'label_map': label_map, 'labels': test_labels})

if __name__ == "__main__":
    args = parse_args()

    config_file = args.config_file
    method = args.method

    if not os.path.exists(config_file):
        raise RuntimeError('Invalid config file')

    if method == "build_flickr":
        #build_flickr requires that data to be in format
        #input_dir
            # class_0
                # image_0
                # image_1
                # ...
            # class_2
                # image_0
                # image_1
                # ...
            # ...

        torch.set_grad_enabled(False)
        create_flickr_data(config_file)
    elif method == "extract_latent":
        #extract_latent requires that data to be in format
        #latent_input_dir
            # texture_0
                # image_0
                # image_1
                # ...
            # texture_2
                # image_0
                # image_1
                # ...
            # ...
        torch.set_grad_enabled(False)
        extract_latent_codes(config_file)
    else:
        raise RuntimeError("Illegal method found: " + method)

