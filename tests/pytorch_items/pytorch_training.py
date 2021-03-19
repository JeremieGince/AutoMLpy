import torch
from torch import nn
from torch.utils.data import Subset,  DataLoader, TensorDataset
import numpy as np
from typing import Dict, Tuple
import logging
import enum
import tqdm


class PhaseType(enum.Enum):
    train = 0
    val = 1
    test = 2


def train_pytorch_network(
        network,
        loaders,
        verbose: bool = False,
        **training_kwargs,
):
    """
    Fit the given network with the given training data.

    Parameters
    ----------
    network: The neural network to fit.
    loaders: The data loaders as a dictionary with keys: {train, valid}.
    verbose: True to show some training stats else False.
    training_kwargs:
        optimiser (torch.optim): The optimizer used to make the weights updates.
        momentum (float): The momentum of the optimiser if the optimiser is not given.
        nesterov (bool): The nesterov of the optimiser if the optimiser is not given.
        use_cuda (bool): True to use cuda device else False.
        scheduler (): A learning rate scheduler.

    Returns
    -------
    last train accuracy, last validation accuracy, the training history.
    """

    training_kwargs.setdefault(
        "optimizer",
        torch.optim.SGD(
            (p for p in network.parameters() if p.requires_grad),
            lr=training_kwargs.get("lr", 1e-3),
            momentum=training_kwargs.get("momentum", 0.9),
            nesterov=training_kwargs.get("nesterov", True),
        )
    )

    training_kwargs.setdefault(
        "criterion",
        torch.nn.CrossEntropyLoss()
    )

    history = []
    nb_epochs = training_kwargs.get("epochs", 5)

    for epoch in range(nb_epochs):
        epoch_logs = {}
        train_logs = execute_phase(network, loaders["train"], PhaseType.train, verbose, **training_kwargs)
        epoch_logs["train"] = train_logs

        if "valid" in loaders:
            val_logs = execute_phase(network, loaders["valid"], PhaseType.val, verbose, **training_kwargs)
            epoch_logs["val"] = val_logs

        history.append(epoch_logs)

    return history


def execute_phase(
    network: nn.Module,
    data_loader: DataLoader,
    phase_type: PhaseType = PhaseType.train,
    verbose: bool = False,
    **kwargs
) -> Dict[str, float]:
    """
    Execute a training phase on a network. The possible phase are {train, val, test}.

    Parameters
    ----------
    network: The model to fit.
    data_loader: The data loader used to make the current training phase.
    phase_type: The phase type in {train, val, test}.
    verbose: True to show some training stats else False.
    kwargs:
        use_cuda (bool): True to use cuda device else False.
        scheduler (): A learning rate scheduler.

    Returns
    -------
    The phase logs.
    """
    if phase_type == PhaseType.train:
        network.train()
    else:
        network.eval()

    if kwargs.get("use_cuda", True):
        device = "cuda"
        if torch.cuda.is_available():
            network.to(device)
    else:
        device = "cpu"

    if "scheduler" in kwargs and kwargs["scheduler"] is not None:
        kwargs["scheduler"].step()

    phase_logs = {"loss": 0, "acc": 0}

    if verbose:
        phase_progress = tqdm.tqdm(range(len(data_loader)), unit="batch")
        phase_progress.set_description_str(f"Phase: {phase_type.name}")
    for j, (inputs, targets) in enumerate(data_loader):
        if device == "cuda":
            if torch.cuda.is_available():
                inputs = inputs.float().to(device)
                targets = targets.to(device)

        batch_logs = execute_batch_training(network, inputs, targets, phase_type, verbose, **kwargs)
        for metric_name, metric in batch_logs.items():
            phase_logs[metric_name] = (j * phase_logs[metric_name] + metric) / (j + 1)

        if verbose:
            phase_progress.update()
            phase_progress.set_postfix_str(' '.join([str(_m)+': '+str(f"{_v:.5f}")
                                                     for _m, _v in phase_logs.items()]))
    if verbose:
        phase_progress.close()
    return phase_logs


def execute_batch_training(
    network: nn.Module,
    inputs,
    targets,
    phase_type: PhaseType = PhaseType.train,
    verbose: bool = False,
    **kwargs
) -> Dict[str, float]:
    """
    Execute a training batch on a network.

    Parameters
    ----------
    network: The model to fit.
    inputs: The inputs of the model.
    targets: The targets of the model.
    phase_type: The phase type in {train, val, test}.
    verbose: True to show some training stats else False.
    kwargs:
        optimiser (torch.optim): The optimizer used to make the weights updates.

    Returns
    -------
    Batch logs as dict.
    """
    network.zero_grad()
    output = network(inputs)

    if verbose:
        logging.debug(f"\n {output}")
    batch_logs = dict(loss=kwargs["criterion"](output, targets))

    if phase_type == PhaseType.train:
        batch_logs["loss"].backward()
        kwargs["optimizer"].step()

    batch_logs['acc'] = np.mean((torch.argmax(output, dim=-1) == targets).cpu().detach().numpy())

    batch_logs["loss"] = batch_logs["loss"].cpu().detach().numpy()
    return batch_logs
