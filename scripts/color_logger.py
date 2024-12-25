import logging
import colorlog

class ColorLogger:
    def __init__(self, name, level=logging.INFO):
        self.counters = {"INFO": 0, "WARNING": 0, "ERROR": 0, "CRITICAL": 0}

        handler = colorlog.StreamHandler()
        handler.setFormatter(colorlog.ColoredFormatter(
            '%(log_color)s[%(levelname)s] %(message)s',
            log_colors={
                'DEBUG': 'blue',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        ))

        self.logger = colorlog.getLogger(name)
        self.logger.addHandler(handler)
        self.logger.setLevel(level)

        self.wrap_log_methods()

    def wrap_log_methods(self):
        original_methods = {
            "info": self.logger.info,
            "warning": self.logger.warning,
            "error": self.logger.error,
            "critical": self.logger.critical,
        }

        for level in original_methods:
            def wrapped_log_method(message, *, level=level):
                self.counters[level.upper()] += 1
                original_methods[level](message)
            setattr(self.logger, level, wrapped_log_method)

    def get_counters(self):
        return self.counters

    def reset_counters(self):
        for key in self.counters:
            self.counters[key] = 0

    def get_logger(self):
        return self.logger