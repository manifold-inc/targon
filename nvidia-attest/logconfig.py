import logging.config
from pythonjsonlogger.json import JsonFormatter
import yaml


class CustomJsonFormatter(JsonFormatter):
    def format(self, record):
        # Add the 'location' field by combining 'pathname' and 'lineno'
        record.location = f"{record.pathname}:{record.lineno}"
        return super().format(record)


def setupLogging():
    with open("logconfig.yaml", "rt") as file:
        config = yaml.safe_load(file.read())
    logging.config.dictConfig(config)
    logger = logging.getLogger("main")
    return logger
