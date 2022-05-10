import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import shutil
import numpy as np

import util
from data.classifier_data_loader import get_data_loader, get_test_transform
from options import TestOptions
from util.diff_augment import DiffAugment
diff_aug_policy = 'color,translation,cutout'

from create_training_data import parse_and_validate_gan_details

methods = ['train', 'infer']
augment_types = ['simple', 'deluxe']
"TODO: move these all to config"
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True, choices=methods)
    parser.add_argument('--label_path', required=True)
    parser.add_argument('--checkpoint_dir', default=None)
    parser.add_argument('--checkpont_save_epoch', type=int, default=10)
    parser.add_argument('--checkpoint_file', default=None)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--num_epochs', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--results_dir', default=None)
    parser.add_argument('--visualize_missed_detections', action='store_true')
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--config_file', default=None)
    parser.add_argument('--ensemble_alpha', type=float, default=0.5)
    parser.add_argument('--augment_type', default='simple', choices=augment_types)
    parser.add_argument('--use_diffaug', action='store_true')

    args = parser.parse_args()
    return args

def build_model(num_classes):
    cnn = models.vgg19(pretrained=True).to("cuda").eval()

    classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 1024),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, num_classes)
    )

    cnn.classifier = classifier.to("cuda")
    
    return cnn

def save_checkpoint(epoch, checkpoint_dir, model):
    path = os.path.join(checkpoint_dir, 'epoch_%d.pkl' % epoch)
    print('Saving checkpoint: ' + path)
    torch.save(model.state_dict(), path)

def train(label_path, checkpoint_dir, lr, num_epochs, batch_size, augment_type, use_diffaug, checkpont_save_epoch):
    label_dict = util.read_file(label_path)
    img_size = label_dict['img_size']
    num_classes = label_dict['num_classes']

    model = build_model(num_classes)
    dataloader = get_data_loader(label_dict['labels'], True, image_size=img_size, batch_size=batch_size, augment_type=augment_type)
    optimizer = optim.Adam(model.parameters(), lr)
    
    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        iteration = 0
        losses = 0
        for batch in dataloader:
            images, labels = batch

            if use_diffaug:
               images = DiffAugment(images, policy=diff_aug_policy)

            images = images.cuda()
            labels = labels.cuda()

            pred = model(images)

            loss = ce_loss(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses += loss.item()

            iteration += 1

        losses = losses / iteration
        print('Epoch [{:4d}] | loss: {:6.4f}'.format(
                epoch, losses))

        # Save the model parameters
        save_epoch = epoch + 1
        if save_epoch % checkpont_save_epoch == 0:
            save_checkpoint(save_epoch, checkpoint_dir, model)

#TODO am I able to batch inputs to the model 
#instead of running 1 by 1?
def get_ensemble_data(image_paths, image_size, gan_opt, gan_model, gan_details):
    ensemble_augment_textures = gan_details['ensemble_augment_textures']
    ensemble_images = []
    ensemble_image_indices = []

    transorm = get_test_transform(image_size)
    
    for texture in ensemble_augment_textures:
        latent_path = os.path.join(gan_details['extracted_latent_code_dir'], texture, 'latent_codes.pth')
        texture_codes = torch.load(latent_path)
        rand_ind = np.random.choice(texture_codes.shape[0])
        texture_code = texture_codes[rand_ind:rand_ind+1]

        for i in range(len(image_paths)):
            image_path = image_paths[i]
            structure_image = util.load_image(gan_opt, image_path)
            structure_code, structure_texture = gan_model(structure_image, command="encode")
            texture_code = util.lerp(structure_texture, texture_code, gan_details['texture_alpha'])
            augmented_image = gan_model(structure_code, texture_code, command="decode")
            augmented_image = transforms.ToPILImage()((augmented_image[0].clamp(-1.0, 1.0) + 1.0) * 0.5)

            augmented_image = transorm(augmented_image)

            ensemble_images.append(augmented_image)
            ensemble_image_indices.append(i)

    return ensemble_images, np.array(ensemble_image_indices)


def infer(label_path, checkpoint_path, results_dir, visualize_missed_detections, ensemble, config_file, ensemble_alpha):
    label_dict = util.read_file(label_path)
    img_size = label_dict['img_size']
    num_classes = label_dict['num_classes']
    label_map = label_dict['label_map']

    model = build_model(num_classes)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    dataloader = get_data_loader(label_dict['labels'], False, image_size=img_size, batch_size=batch_size, return_path=True)

    if visualize_missed_detections:
        visualize_missed_detections_dir = os.path.join(results_dir, 'missed_detections')
        for i in range(num_classes):
            class_name = label_map[i]
            os.makedirs(os.path.join(visualize_missed_detections_dir, class_name), exist_ok=True)

    if ensemble:
        torch.set_grad_enabled(False)
        config = util.read_yaml(config_file)
        gan_opt, gan_model = util.get_opt_and_model(opt = TestOptions().parse(name_req=False))
        gan_augment, gan_details = parse_and_validate_gan_details(config, label_map, False)
        if not gan_augment:
            raise RuntimeError('gan_augment must be True when ensembling')

    total_corr = 0.0
    total = 0.0
    for batch in dataloader:
        images, labels, image_paths = batch
        images = images.cuda()
        labels = labels.cuda()

        if not ensemble:
            pred = model(images)
            pred_labels = torch.argmax(pred, dim=1)
        else:
            ensemble_images, ensemble_image_indices = get_ensemble_data(image_paths, config['img_size'], gan_opt, gan_model, gan_details)
            ensemble_images = torch.stack(ensemble_images).cuda()

            orig_preds = model(images)

            ensemble_preds = []
            ensemble_start_index = 0
            while ensemble_start_index < ensemble_image_indices.shape[0]:
                ensemble_end_index = np.min([ensemble_start_index + batch_size, ensemble_image_indices.shape[0]])
                ensemble_preds.append(model(ensemble_images[ensemble_start_index:ensemble_end_index]))

                ensemble_start_index = ensemble_end_index

            ensemble_preds = torch.cat(ensemble_preds)

            preds = []
            for image_index in range(len(image_paths)):
                ensemble_inds = np.where(ensemble_image_indices == image_index)
                ensemble_image_preds = ensemble_preds[ensemble_inds]

                image_preds = orig_preds[image_index]*(1-ensemble_alpha) + torch.mean(ensemble_image_preds, dim=0)*(ensemble_alpha)
                preds.append(image_preds)

            preds = torch.stack(preds)
            pred_labels = torch.argmax(preds, dim=1)

        total_corr += (pred_labels == labels).sum()
        total += labels.shape[0]

        if visualize_missed_detections:
            for i in range(images.shape[0]):
                if pred_labels[i] == labels[i]:
                    continue
                    
                correct_class = label_map[labels[i].item()]
                predicted_class = label_map[pred_labels[i].item()]

                dest_dir = os.path.join(visualize_missed_detections_dir, correct_class)
                dest_path = os.path.join(dest_dir, predicted_class + '_' + os.path.basename(image_paths[i]))
                shutil.copyfile(image_paths[i], dest_path)

    accuracy = total_corr / total
    print('Accuracy is: ', accuracy)

    util.write_json(os.path.join(results_dir, 'accuracy.json'), {'accuracy': accuracy.item()})

if __name__ == "__main__":
    args = parse_args()

    method = args.method
    label_path = args.label_path
    checkpoint_dir = args.checkpoint_dir
    checkpont_save_epoch = args.checkpont_save_epoch
    checkpoint_file = args.checkpoint_file
    lr = args.lr
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    results_dir = args.results_dir
    visualize_missed_detections = args.visualize_missed_detections
    ensemble = args.ensemble
    config_file = args.config_file
    ensemble_alpha = args.ensemble_alpha
    augment_type = args.augment_type
    use_diffaug = args.use_diffaug

    if not os.path.exists(label_path):
        raise RuntimeError('Invalid label path')

    if method == "train":
        if checkpoint_dir is None:
            raise RuntimeError('checkpoint_dir required for training')

        os.makedirs(checkpoint_dir, exist_ok=True)

        train(label_path, checkpoint_dir, lr, num_epochs, batch_size, augment_type, use_diffaug, checkpont_save_epoch)
    elif method == 'infer':
        if checkpoint_file is None:
            raise RuntimeError('checkpoint_file required for inference')

        if results_dir is None:
            raise RuntimeError('results_dir required for inference')

        if (ensemble) and (config_file is None):
            raise RuntimeError('config_file must be specified if ensemble is True')

        os.makedirs(results_dir, exist_ok=True)

        infer(label_path, checkpoint_file, results_dir, visualize_missed_detections, ensemble, config_file, ensemble_alpha)
    else:
        raise RuntimeError("Illegal method found: " + method)