from src.optimizer import HpOptimizer
from tests.pytorch_items.pytorch_models import CifarNet, MnistNet, CifarNetBatchNorm
import poutyne as pt
import time
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Subset,  DataLoader, TensorDataset
from torchvision.transforms import ToTensor
from typing import Dict, Union, List, Iterable, Hashable, Tuple
import numpy as np
from torchvision.datasets.mnist import MNIST


def get_MNIST_dataloaders(seed: int = 42):
    train_split_ratio = 0.8
    batch_size = 64

    np.random.seed(seed)

    full_train_dataset = MNIST('./datasets/', train=True, download=True, transform=ToTensor())
    test_dataset = MNIST('./datasets/', train=False, download=True, transform=ToTensor())

    indices = list(range(len(full_train_dataset)))
    np.random.shuffle(indices)

    split_index = np.floor(train_split_ratio * len(full_train_dataset)).astype(int)

    train_indices = indices[:split_index]
    train_dataset = Subset(full_train_dataset, train_indices)

    valid_indices = indices[split_index:]
    valid_dataset = Subset(full_train_dataset, valid_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2)

    return dict(train=train_loader, valid=valid_loader, test=test_loader)


class PoutyneMNISTHpOptimizer(HpOptimizer):
    def build_model(self, **hp) -> object:
        net = MnistNet()
        optimizer = optim.SGD(net.parameters(),
                              lr=hp.get("learning_rate", 1e-3),
                              momentum=hp.get("momentum", 0.9),
                              nesterov=hp.get("nesterov", True))
        model = pt.Model(net, optimizer, 'cross_entropy', batch_metrics=['accuracy'])
        if torch.cuda.is_available():
            model.cuda()
        return model

    def fit_model_(
            self,
            model: pt.Model,
            X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
            y: Union[np.ndarray, torch.Tensor],
            verbose=False,
            **hp
    ) -> object:
        if hp.get("pre_normalized", True):
            X = X/torch.max(X)
        history = model.fit_generator(
            DataLoader(
                TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)),
                batch_size=hp.get("batch_size", 64),
                num_workers=2,
                shuffle=True
            ),
            epochs=hp.get("epochs", 1),
            verbose=verbose
        )
        return model

    def score(
            self,
            model: pt.Model,
            X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
            y: Union[np.ndarray, torch.Tensor],
            **hp
    ) -> Tuple[float, float]:
        if hp.get("pre_normalized", True):
            X = X/torch.max(X)
        test_loss, test_acc = model.evaluate_generator(
            DataLoader(
                TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)),
                batch_size=hp.get("batch_size", 64),
                num_workers=2
            )
        )
        return test_acc/100, 0.0


class PoutyneCifar10HpOptimizer(HpOptimizer):
    def build_model(self, **hp) -> pt.Model:
        if hp.get("use_batchnorm", True):
            net = CifarNetBatchNorm()
        else:
            net = CifarNet()
        optimizer = optim.SGD(net.parameters(),
                              lr=hp.get("learning_rate", 1e-3),
                              momentum=hp.get("momentum", 0.0),
                              nesterov=hp.get("nesterov", False))
        model = pt.Model(net, optimizer, 'cross_entropy', batch_metrics=['accuracy'])
        if torch.cuda.is_available():
            model.cuda()
        return model

    def fit_model_(
            self,
            model: pt.Model,
            X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
            y: Union[np.ndarray, torch.Tensor],
            verbose=False,
            **hp
    ) -> pt.Model:
        if hp.get("pre_normalized", True):
            X = X/torch.max(X)
        history = model.fit_generator(
            DataLoader(
                TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)),
                batch_size=hp.get("batch_size", 64),
                num_workers=2,
                shuffle=True
            ),
            epochs=hp.get("epochs", 1),
            verbose=verbose
        )
        return model

    def score(
            self,
            model: pt.Model,
            X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
            y: Union[np.ndarray, torch.Tensor],
            **hp
    ) -> Tuple[float, float]:
        if hp.get("pre_normalized", True):
            X = X/torch.max(X)
        test_loss, test_acc = model.evaluate_generator(
            DataLoader(
                TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)),
                batch_size=hp.get("batch_size", 64),
                num_workers=2
            )
        )
        return test_acc/100, 0.0


if __name__ == '__main__':
    from tests.pytorch_items.pytorch_datasets import get_MNIST_X_y, get_Cifar10_X_y

    hp = dict(
        epochs=15,
        batch_size=64,
        learning_rate=0.1,
        nesterov=True,
        momentum=0.9,
        use_batchnorm=True,
        pre_normalized=True,
    )
    X_y_dict = get_Cifar10_X_y()
    opt = PoutyneCifar10HpOptimizer()
    model = opt.build_model(**hp)
    opt.fit_model_(
        model,
        X_y_dict["train"]["x"],
        X_y_dict["train"]["y"],
        verbose=True,
        **hp
    )

    test_acc, _ = opt.score(
        model,
        X_y_dict["test"]["x"],
        X_y_dict["test"]["y"],
        **hp
    )
    print(test_acc)
