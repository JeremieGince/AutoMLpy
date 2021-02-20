
def logs_file_setup(file: str):
    import logging
    import os
    import sys
    import time
    from datetime import date

    today = date.today()
    timestamp = str(time.time()).replace('.', '')
    logs_dir = f"logs/logs-{today.strftime('%d-%m-%Y')}"
    logs_file = f'{logs_dir}/{os.path.splitext(os.path.basename(file))[0]}-{timestamp}.log'
    os.makedirs(logs_dir, exist_ok=True)
    logging.basicConfig(filename=logs_file, filemode='w+', level=logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    logging.getLogger().addHandler(sh)


def log_device_setup():
    import logging
    import torch
    import sys
    from subprocess import check_output

    logging.info(f'__Python VERSION:{sys.version}')
    logging.info(f'__pyTorch VERSION:{torch.__version__}')
    logging.info(f'__CUDA VERSION:\n{check_output(["nvcc", "--version"]).decode("utf-8")}')
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
