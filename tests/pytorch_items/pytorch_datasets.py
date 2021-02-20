import os
import math
import random
import numpy as np
import torch
from torch.utils.data import Subset, Dataset, DataLoader
from torchvision.transforms import ToTensor, ConvertImageDtype, Compose
from torchvision import transforms
from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10

BASE_PATH = '~/tests/pytorch_datasets/'


def get_MNIST_datasets(seed: int = 42, path=os.path.join(BASE_PATH, 'mnist'), **kwargs):
    train_split_ratio = 0.8

    np.random.seed(seed)

    mnist_transforms = Compose(
        [
            ToTensor(),
            ConvertImageDtype(torch.float),
            transforms.Lambda(lambda x: x/255.),
            # ToTensor(),
         ]
    )

    full_train_dataset = MNIST(path, train=True, download=True, transform=mnist_transforms)
    test_dataset = MNIST(path, train=False, download=True, transform=mnist_transforms)

    indices = list(range(len(full_train_dataset)))
    np.random.shuffle(indices)

    split_index = np.floor(train_split_ratio * len(full_train_dataset)).astype(int)

    train_indices = indices[:split_index]
    train_dataset = Subset(full_train_dataset, train_indices)

    valid_indices = indices[split_index:]
    valid_dataset = Subset(full_train_dataset, valid_indices)

    return dict(train=train_dataset, valid=valid_dataset, test=test_dataset)


def get_MNIST_dataloaders(seed: int = 42, path=os.path.join(BASE_PATH, 'mnist'), **kwargs):
    batch_size = 64
    mnist_datasets = get_MNIST_datasets(seed, path)

    train_loader = DataLoader(mnist_datasets["train"], batch_size=batch_size, num_workers=2, shuffle=True)
    valid_loader = DataLoader(mnist_datasets["valid"], batch_size=batch_size, num_workers=2)
    test_loader = DataLoader(mnist_datasets["test"], batch_size=batch_size, num_workers=2)

    return dict(train=train_loader, valid=valid_loader, test=test_loader)


def get_MNIST_X_y(**kwargs):
    # TODO: optimize
    datasets = get_MNIST_datasets(**kwargs)
    X_y_dict = {phase: dict(x=[], y=[]) for phase in datasets}
    for phase, dataset in datasets.items():
        for x, y in dataset:
            # x, y = datasets[phase].transforms(x, y)
            X_y_dict[phase]["x"].append(x)
            X_y_dict[phase]["y"].append(y)
    for phase in X_y_dict:
        X_y_dict[phase]["x"] = torch.stack(X_y_dict[phase]["x"], dim=0)
        X_y_dict[phase]["y"] = torch.LongTensor(X_y_dict[phase]["y"])
    return X_y_dict


def get_Cifar10_datasets(seed: int = 42, path=os.path.join(BASE_PATH, 'cifar10'), **kwargs):
    train_split_ratio = 0.8

    np.random.seed(seed)

    cifar10_transforms = Compose(
        [
            ToTensor(),
            ConvertImageDtype(torch.float),
            transforms.Lambda(lambda x: x/255.),
            # ToTensor(),
         ]
    )

    full_train_dataset = CIFAR10(path, train=True, download=True, transform=cifar10_transforms)
    test_dataset = CIFAR10(path, train=False, download=True, transform=cifar10_transforms)

    indices = list(range(len(full_train_dataset)))
    np.random.shuffle(indices)

    split_index = np.floor(train_split_ratio * len(full_train_dataset)).astype(int)

    train_indices = indices[:split_index]
    train_dataset = Subset(full_train_dataset, train_indices)

    valid_indices = indices[split_index:]
    valid_dataset = Subset(full_train_dataset, valid_indices)

    return dict(train=train_dataset, valid=valid_dataset, test=test_dataset)


def get_Cifar10_dataloaders(seed: int = 42, path=os.path.join(BASE_PATH, 'cifar10'), **kwargs):
    batch_size = 64
    cifar10_datasets = get_Cifar10_datasets(seed, path)

    train_loader = DataLoader(cifar10_datasets["train"], batch_size=batch_size, num_workers=2, shuffle=True)
    valid_loader = DataLoader(cifar10_datasets["valid"], batch_size=batch_size, num_workers=2)
    test_loader = DataLoader(cifar10_datasets["test"], batch_size=batch_size, num_workers=2)

    return dict(train=train_loader, valid=valid_loader, test=test_loader)


def get_Cifar10_X_y(**kwargs):
    # TODO: optimize
    datasets = get_Cifar10_datasets(**kwargs)
    X_y_dict = {phase: dict(x=[], y=[]) for phase in datasets}
    for phase, dataset in datasets.items():
        for x, y in dataset:
            # x, y = datasets[phase].transforms(x, y)
            X_y_dict[phase]["x"].append(x)
            X_y_dict[phase]["y"].append(y)
    for phase in X_y_dict:
        X_y_dict[phase]["x"] = torch.stack(X_y_dict[phase]["x"], dim=0)
        X_y_dict[phase]["y"] = torch.LongTensor(X_y_dict[phase]["y"])
    return X_y_dict


class SpiralDataset(Dataset):
    """
    Un jeu de données synthétique de spiral pour PyTorch.
    Args:
        n_points (int): Le nombre de point désiré dans le jeu de données
        noise (float): Quantité de bruit désiré dans le jeu de données
    """

    def __init__(self, n_points=1000, noise=0.2):
        self.points = torch.Tensor(n_points, 7)
        self.labels = torch.LongTensor(n_points)

        n_positive = n_points // 2
        n_negative = n_points = n_positive

        for i, point in enumerate(self._gen_spiral_points(n_positive, 0, noise)):
            self.points[i], self.labels[i] = point, 1

        for i, point in enumerate(self._gen_spiral_points(n_negative, math.pi, noise)):
            self.points[i + n_positive] = point
            self.labels[i + n_positive] = 0

    def _gen_spiral_points(self, n_points, delta_t, noise):
        for i in range(n_points):
            r = i / n_points * 5
            t = 1.75 * i / n_points * 2 * math.pi + delta_t
            x = r * math.sin(t) + random.uniform(-1, 1) * noise
            y = r * math.cos(t) + random.uniform(-1, 1) * noise
            yield torch.Tensor([x, y, x**2, y**2, x * y, math.sin(x), math.sin(y)])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.points[i], self.labels[i]

    def to_numpy(self):
        return self.points.numpy(), self.labels.numpy()


if __name__ == '__main__':
    get_MNIST_datasets()
    x_y_dict = get_MNIST_X_y()
    print(x_y_dict["train"]["x"].shape, x_y_dict["train"]["y"].shape)
    print(x_y_dict["test"]["x"].shape, x_y_dict["test"]["y"].shape)