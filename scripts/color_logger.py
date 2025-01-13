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
        levels = ["info", "warning", "error", "critical"]

        for level in levels:
            original_method = getattr(self.logger, level)

            def wrapped_log_method(message, orig_method=original_method, log_level=level):
                self.counters[log_level.upper()] += 1
                orig_method(message)

            setattr(self.logger, level, wrapped_log_method)

    def get_counters(self):
        return self.counters

    def get_logger(self):
        return self.logger