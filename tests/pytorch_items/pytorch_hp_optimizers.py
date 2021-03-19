import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Subset,  DataLoader, TensorDataset
from torchvision.transforms import ToTensor
from typing import Union, Tuple
import numpy as np
from torchvision.datasets.mnist import MNIST

from AutoMLpy.optimizers.optimizer import HpOptimizer
from tests.pytorch_items.pytorch_models import CifarNet, MnistNet, CifarNetBatchNorm
from tests.pytorch_items.pytorch_training import train_pytorch_network


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


class TorchMNISTHpOptimizer(HpOptimizer):
    def build_model(self, **hp) -> torch.nn.Module:
        model = MnistNet()
        if torch.cuda.is_available():
            model.cuda()
        return model

    def fit_model_(
            self,
            model: torch.nn.Module,
            X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
            y: Union[np.ndarray, torch.Tensor],
            verbose=False,
            **hp
    ) -> object:
        if hp.get("pre_normalized", True):
            X = X/torch.max(X)

        optimizer = optim.SGD(model.parameters(),
                              lr=hp.get("learning_rate", 1e-3),
                              momentum=hp.get("momentum", 0.9),
                              nesterov=hp.get("nesterov", True))

        train_pytorch_network(
            model,
            loaders=dict(
                train=DataLoader(
                    TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)),
                    batch_size=hp.get("batch_size", 64),
                    num_workers=2,
                    shuffle=True
                )
            ),
            verbose=verbose,
            optimizer=optimizer,
            **hp
        )

        return model

    def score(
            self,
            model: torch.nn.Module,
            X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
            y: Union[np.ndarray, torch.Tensor],
            **hp
    ) -> float:
        if hp.get("pre_normalized", True):
            X = X/torch.max(X)

        model_device = next(model.parameters()).device
        if isinstance(X, torch.Tensor):
            X = X.float().to(model_device)
            y = y.to(model_device)
        test_acc = np.mean((torch.argmax(model(X), dim=-1) == y).cpu().detach().numpy())
        return test_acc


class TorchCifar10HpOptimizer(HpOptimizer):
    def build_model(self, **hp) -> torch.nn.Module:
        if hp.get("use_batchnorm", True):
            model = CifarNetBatchNorm()
        else:
            model = CifarNet()

        if torch.cuda.is_available():
            model.cuda()
        return model

    def fit_model_(
            self,
            model: torch.nn.Module,
            X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
            y: Union[np.ndarray, torch.Tensor],
            verbose=False,
            **hp
    ) -> torch.nn.Module:
        if hp.get("pre_normalized", True):
            X = X/torch.max(X)

        optimizer = optim.SGD(model.parameters(),
                              lr=hp.get("learning_rate", 1e-3),
                              momentum=hp.get("momentum", 0.9),
                              nesterov=hp.get("nesterov", True))

        train_pytorch_network(
            model,
            loaders=dict(
                train=DataLoader(
                    TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)),
                    batch_size=hp.get("batch_size", 64),
                    num_workers=2,
                    shuffle=True
                )
            ),
            verbose=verbose,
            optimizer=optimizer,
            **hp
        )
        return model

    def score(
            self,
            model: torch.nn.Module,
            X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
            y: Union[np.ndarray, torch.Tensor],
            **hp
    ) -> float:
        if hp.get("pre_normalized", True):
            X = X/torch.max(X)

        model_device = next(model.parameters()).device
        if isinstance(X, torch.Tensor):
            X = X.float().to(model_device)
            y = y.to(model_device)
        test_acc = np.mean((torch.argmax(model(X), dim=-1) == y).cpu().detach().numpy())
        return test_acc


if __name__ == '__main__':
    from tests.pytorch_items.pytorch_datasets import get_torch_Cifar10_X_y

    hp = dict(
        epochs=15,
        batch_size=64,
        learning_rate=1e-3,
        nesterov=True,
        momentum=0.9,
        use_batchnorm=True,
        pre_normalized=False,
    )
    X_y_dict = get_torch_Cifar10_X_y()
    opt = TorchCifar10HpOptimizer()
    model_ = opt.build_model(**hp)
    opt.fit_model_(
        model_,
        X_y_dict["train"]["x"],
        X_y_dict["train"]["y"],
        verbose=True,
        **hp
    )

    test_acc = opt.score(
        model_.cpu(),
        X_y_dict["test"]["x"],
        X_y_dict["test"]["y"],
        **hp
    )
    print(test_acc)
