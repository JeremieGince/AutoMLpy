from src import logging_tools
import logging

if __name__ == '__main__':
    logging_tools.logs_file_setup(__file__, level=logging.INFO)
    logging_tools.log_device_setup()
    logging.info(r"More info about the package at: https://github.com/JeremieGince/AutoMLpy")

