import logging
import os
from os.path import join, dirname, exists

def init_logger(config):
    """
    A logger that can show a message on standard output and write it into the
    file named `filename` simultaneously.
    All the message that you want to log MUST be str.

    Args:
        config (config.configuration.Config): An instance object of Config, used to record parameter information.
    """
    

    logfilepath =config['log_file'] if config['log_file'] else config['log_path']
    filefmt = "%(asctime)-15s %(levelname)s %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = "%(asctime)-15s %(levelname)s %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = logging.Formatter(sfmt, sdatefmt)
    if config['state'] is None or config['state'].lower() == 'info':
        level = logging.INFO
    elif config['state'].lower() == 'debug':
        level = logging.DEBUG
    elif config['state'].lower() == 'error':
        level = logging.ERROR
    elif config['state'].lower() == 'warning':
        level = logging.WARNING
    elif config['state'].lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO
    
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)
    handlers=[sh]

    if config['save_train'] == True:
        if not exists(dirname(join(os.getcwd(), logfilepath))):
            os.makedirs(dirname(logfilepath))
        fh = logging.FileHandler(logfilepath)
        fh.setLevel(level)
        fh.setFormatter(fileformatter)
        handlers.append(fh)


    logging.basicConfig(
        level=level,
        handlers=handlers
    )