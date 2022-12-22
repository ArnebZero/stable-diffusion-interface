import os
import logging
from logging.handlers import TimedRotatingFileHandler


class CustomFormatter(logging.Formatter):
    format_pattern = "[{level} %(asctime)s %(pathname)s:%(lineno)d] %(message)s"

    FORMATS = {
        logging.DEBUG: format_pattern.format(level="D"),
        logging.INFO: format_pattern.format(level="I"),
        logging.ERROR: format_pattern.format(level="E"),
        logging.WARNING: format_pattern.format(level="W"),
        logging.CRITICAL: format_pattern.format(level="C"),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%y%m%d %H:%M:%S")

        if "pathname" in record.__dict__ and isinstance(record.pathname, str):
            # truncate the pathname
            filename = os.path.basename(record.pathname)
            filename = os.path.splitext(filename)[0]

            if len(filename) > 20:
                filename = f"{filename[:3]}~{filename[-16:]}"
            record.pathname = filename

        return formatter.format(record)


def get_logger(name=None, log_file_path="logs/app.log", backup_count=24):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    log_handler = TimedRotatingFileHandler(log_file_path, backupCount=backup_count)
    log_handler.setLevel(logging.DEBUG)

    console_handler.setFormatter(CustomFormatter())
    log_handler.setFormatter(CustomFormatter())

    logger.addHandler(console_handler)
    logger.addHandler(log_handler)

    return logger


def get_file_handler(log_file_path="logs/app.log", backup_count=24):
    log_handler = TimedRotatingFileHandler(log_file_path, backupCount=backup_count)
    log_handler.setLevel(logging.DEBUG)

    log_handler.setFormatter(CustomFormatter())

    return log_handler

