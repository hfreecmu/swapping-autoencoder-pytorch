import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import shutil

import util
from data.classifier_data_loader import get_data_loader

methods = ['train', 'infer']
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

def train(label_path, checkpoint_dir, lr, num_epochs, batch_size, checkpont_save_epoch):
    label_dict = util.read_file(label_path)
    img_size = label_dict['img_size']
    num_classes = label_dict['num_classes']

    model = build_model(num_classes)
    dataloader = get_data_loader(label_dict['labels'], True, image_size=img_size, batch_size=batch_size)
    optimizer = optim.Adam(model.parameters(), lr)
    
    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        iteration = 0
        losses = 0
        for batch in dataloader:
            images, labels = batch
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

def infer(label_path, checkpoint_path, results_dir, visualize_missed_detections):
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

    total_corr = 0.0
    total = 0.0
    for batch in dataloader:
        images, labels, image_paths = batch
        images = images.cuda()
        labels = labels.cuda()

        pred = model(images)
        pred_labels = torch.argmax(pred, dim=1)

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

    if not os.path.exists(label_path):
        raise RuntimeError('Invalid label path')

    if method == "train":
        if checkpoint_dir is None:
            raise RuntimeError('checkpoint_dir required for training')

        os.makedirs(checkpoint_dir, exist_ok=True)

        train(label_path, checkpoint_dir, lr, num_epochs, batch_size, checkpont_save_epoch)
    elif method == 'infer':
        if checkpoint_file is None:
            raise RuntimeError('checkpoint_file required for inference')

        if results_dir is None:
            raise RuntimeWarning('results_dir required for inference')

        os.makedirs(results_dir, exist_ok=True)

        infer(label_path, checkpoint_file, results_dir, visualize_missed_detections)
    else:
        raise RuntimeError("Illegal method found: " + method)