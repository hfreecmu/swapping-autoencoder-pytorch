import argparse
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import cm

results_type_choices = ['dim_vis', 'loss_vis']
method_choices = ['pca', 'tsne']
dim_choices = [2, 3]

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_type', required=True, choices=results_type_choices)

    #required for dim_vis
    parser.add_argument('--texture_dir', default=None)
    parser.add_argument('--method', default='pca', choices=method_choices)
    parser.add_argument('--dim', type=int, default=2, choices=dim_choices)
    parser.add_argument('--texture_types', default='snow;autumn_trees;night;sunset')

    #required for loss_vis
    parser.add_argument('--train_file', default=None)

    args = parser.parse_args()
    return args

def plot_tsne(latent_codes, latent_labels, latent_label_dict, dim):
    assert (dim == 2) or (dim == 3)

    tsne = TSNE(dim, verbose=1)
    tsne_proj = tsne.fit_transform(latent_codes)
    cmap = cm.get_cmap('tab10')
    fig = plt.figure()
    if dim == 2:
        ax = fig.add_subplot()
    else:
        ax = fig.add_subplot(projection='3d')

    num_categories = len(latent_label_dict)

    for lab in range(num_categories):
        indices = latent_labels == lab

        if dim == 2:
            ax.scatter(tsne_proj[indices, 0],
                tsne_proj[indices, 1],
                c=np.array(cmap(lab)).reshape(1, 4),
                label=latent_label_dict[lab],
                alpha=0.5)
        else:
            ax.scatter(tsne_proj[indices, 0],
                tsne_proj[indices, 1],
                tsne_proj[indices, 2],
                c=np.array(cmap(lab)).reshape(1, 4),
                label=latent_label_dict[lab],
                alpha=0.5)
    ax.legend(fontsize='large', markerscale=2)
    plt.show()

def plot_pca(latent_codes, latent_labels, latent_label_dict, dim):
    assert (dim == 2) or (dim == 3)

    pca = PCA(n_components=dim)
    components = pca.fit_transform(latent_codes)
    cmap = cm.get_cmap('tab10')
    fig = plt.figure()
    if dim == 2:
        ax = fig.add_subplot()
    else:
        ax = fig.add_subplot(projection='3d')

    num_categories = len(latent_label_dict)

    for lab in range(num_categories):
        indices = latent_labels == lab

        if dim == 2:
            ax.scatter(components[indices, 0],
                components[indices, 1],
                c=np.array(cmap(lab)).reshape(1, 4),
                label=latent_label_dict[lab],
                alpha=0.5)
        else:
            ax.scatter(components[indices, 0],
                components[indices, 1],
                components[indices, 2],
                c=np.array(cmap(lab)).reshape(1, 4),
                label=latent_label_dict[lab],
                alpha=0.5)
    ax.legend(fontsize='large', markerscale=2)
    plt.show()

def visualize_latent_codes(texture_dir, method, dim, texture_types):
    latent_codes = []
    latent_labels = []
    latent_label_dict = dict()
    for dirname in os.listdir(texture_dir):
        if dirname not in texture_types:
            continue

        subdir = os.path.join(texture_dir, dirname)
        if not os.path.isdir(subdir):
            continue

        latent_file = os.path.join(subdir, 'latent_codes.pth')

        if not os.path.exists(latent_file):
            continue

        class_codes = torch.load(latent_file)

        label = len(latent_label_dict)
        latent_label_dict[label] = dirname

        latent_codes.append(class_codes)
        latent_labels.extend([label]*class_codes.shape[0])

    latent_codes = torch.cat(latent_codes).detach().cpu().numpy()
    latent_labels = np.array(latent_labels)
        
    if method == 'tsne':
        plot_tsne(latent_codes, latent_labels, latent_label_dict, dim)
    elif method == 'pca':
        plot_pca(latent_codes, latent_labels, latent_label_dict, dim)

def visualize_loss(train_file):
    epochs = []
    losses = []

    with open(train_file) as f:
        lines = f.readlines()
        for ind, line in enumerate(lines):
            if not line.startswith('Epoch'):
                continue

            #remove spaces
            line = line.replace(' ', '')

            epoch = line.split('[')[1].split(']')[0]
            epoch = int(epoch)

            loss = line.split(':')[1]
            loss = float(loss)

            epochs.append(epoch)
            losses.append(loss)
    
    epochs = np.array(epochs)
    losses = np.array(losses)

    data = np.zeros((epochs.shape[0], 2))
    data[:, 0] = epochs
    data[:, 1] = losses

    base_dir = os.path.dirname(train_file)
    data_path = os.path.join(base_dir, 'train_info.npy')

    np.save(data_path, data)

    fig_path = os.path.join(base_dir, 'training_loss.png')

    plt.plot(epochs, losses, 'b')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.savefig(fig_path)

if __name__ == "__main__":
    args = parse_args()

    results_type = args.results_type
    texture_dir = args.texture_dir
    method = args.method
    dim = args.dim
    texture_types = args.texture_types
    texture_types = texture_types.split(';')
    train_file = args.train_file

    if results_type == 'dim_vis':
        if texture_dir is None:
            raise RuntimeError('texture_dir must be specified if results_type is dim_vis')

        if (not os.path.exists(texture_dir)) or (not os.path.isdir(texture_dir)):
            raise RuntimeError('Invalid texture dir')

        visualize_latent_codes(texture_dir, method, dim, texture_types)
    elif results_type == 'loss_vis':
        if train_file is None:
            raise RuntimeError('Invalid train_file')

        if not os.path.isfile(train_file):
            raise RuntimeError('Invalid train_file')

        visualize_loss(train_file)
    else:
        raise RuntimeError('Invalid results_type: ' + results_type)

    
