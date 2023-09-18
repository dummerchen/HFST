import logging
from utils import mkdir

# --------------------------------------------
# logger
# --------------------------------------------
def get_logger(logger_name, log_path='default_logger.log'):
    """ set up logger
    modified by Kai Zhang (github: https://github.com/cszn)
    """
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():
        print('LogHandlers exist!')
    else:
        print('LogHandlers setup!')
        level = logging.INFO
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d : %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(formatter)
        logger.setLevel(level)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger
