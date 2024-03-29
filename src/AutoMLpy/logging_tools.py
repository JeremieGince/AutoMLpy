import enum
import logging


class DeepLib(enum.Enum):
    Null = -1
    Pytorch = 0
    Tensorflow = 1
    SkLearn = 2


def logs_file_setup(file: str, level=logging.INFO, root_logs_dir: str = "./"):
    import os
    import sys
    import time
    from datetime import date

    today = date.today()
    timestamp = str(time.time()).replace('.', '')
    logs_dir = f"{root_logs_dir}/logs/logs-{today.strftime('%d-%m-%Y')}"
    logs_file = f'{logs_dir}/{os.path.splitext(os.path.basename(file))[0]}-{timestamp}.log'
    os.makedirs(logs_dir, exist_ok=True)
    logging.basicConfig(filename=logs_file, filemode='w+', level=level)
    sh = logging.StreamHandler(sys.stdout)
    logging.getLogger().addHandler(sh)
    logging.info(f"Logs file at: {logs_file}\n")


def log_device_setup(deepLib: DeepLib = DeepLib.Null):
    import sys
    from .version import __version__
    import psutil
    import multiprocessing

    logging.info(f'__Python VERSION: {sys.version}')
    logging.info(f"AutoMLpy.version: {__version__}")
    logging.info(f"Number of available cores: {psutil.cpu_count(logical=False)}.")
    logging.info(f"Number of available logical processors: {multiprocessing.cpu_count()}.")

    setup_func = {
        DeepLib.Null: lambda: None,
        DeepLib.Pytorch: log_pytorch_device_setup,
        DeepLib.Tensorflow: log_tensorflow_device_setup,
        DeepLib.SkLearn: log_sklearn_device_setup,
    }
    setup_func[deepLib]()


def log_pytorch_device_setup():
    from subprocess import check_output
    import torch

    logging.info(f'__pyTorch VERSION:{torch.__version__}')
    try:
        logging.info(f'__CUDA VERSION:\n{check_output(["nvcc", "--version"]).decode("utf-8")}')
    except FileNotFoundError:
        logging.info('__CUDA VERSION:Not Found')
    logging.info(f'__CUDNN VERSION:{torch.backends.cudnn.version()}')
    logging.info(f'__Number CUDA Devices:{torch.cuda.device_count()}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"\n{'-' * 25}\nDEVICE: {device}\n{'-' * 25}\n")

    # Additional Info when using cuda
    if device.type == 'cuda':
        logging.info(torch.cuda.get_device_name(0))
        logging.info('Memory Usage:')
        logging.info(f'Allocated: {round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)} GB')
        logging.info(f'Cached:   {round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)} GB')
        logging.info(f"Memory summary: \n{torch.cuda.memory_summary()}")


def log_tensorflow_device_setup():
    import tensorflow as tf
    from subprocess import check_output
    logging.info(f'__tensorflow VERSION:{tf.__version__}')
    try:
        logging.info(f'__CUDA VERSION:\n{check_output(["nvcc", "--version"]).decode("utf-8")}')
    except FileNotFoundError:
        logging.info('__CUDA VERSION:Not Found')
    try:
        logging.info(f'__nvidia-smi:\n{check_output(["nvidia - smi", ]).decode("utf-8")}')
    except FileNotFoundError:
        logging.info('__nvidia-smi: Not Found')
    physical_devices = tf.config.list_physical_devices('GPU')
    logging.info(f"physical_devices: {physical_devices}")
    logical_devices = tf.config.list_logical_devices('GPU')
    logging.info(f"logical_devices: {logical_devices}")


def log_sklearn_device_setup():
    import sklearn
    logging.info(f'__sklearn VERSION:{sklearn.__version__}')

