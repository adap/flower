"""Logger functionality."""

import logging

import coloredlogs


class Logger:
    """Logger class to be used by all modules in the project."""

    log_format = (
        "[%(asctime)s] (%(process)s) {%(filename)s:%(lineno)d}"
        " %(levelname)s - %(message)s"
    )
    log_level = None

    @classmethod
    def setup_logging(cls, loglevel="INFO", logfile=""):
        """Stateful setup of the logging infrastructure.

        :param loglevel: log level to be used
        :param logfile: file to log to
        """
        cls.registered_loggers = {}
        cls.log_level = loglevel
        numeric_level = getattr(logging, loglevel.upper(), None)

        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {loglevel}")
        if logfile:
            logging.basicConfig(
                handlers=[logging.FileHandler(logfile), logging.StreamHandler()],
                level=numeric_level,
                format=cls.log_format,
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        else:
            logging.basicConfig(
                level=numeric_level,
                format=cls.log_format,
                datefmt="%Y-%m-%d %H:%M:%S",
            )

    @classmethod
    def get(cls, logger_name="default"):
        """Get logger instance.

        :param logger_name: name of the logger
        :return: logger instance
        """
        if logger_name in cls.registered_loggers:
            return cls.registered_loggers[logger_name]

        return cls(logger_name)

    def __init__(self, logger_name="default"):
        """Initialise logger not previously registered.

        :param logger_name: name of the logger
        """
        if logger_name in self.registered_loggers:
            raise ValueError(
                f"Logger {logger_name} already exists. "
                f'Call with Logger.get("{logger_name}")'
            )

        self.name = logger_name
        self.logger = logging.getLogger(self.name)
        self.registered_loggers[self.name] = self.logger
        coloredlogs.install(
            level=self.log_level,
            logger=self.logger,
            fmt=self.log_format,
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        self.warn = self.warning

    def log(self, loglevel, msg):
        """Log message.

        :param loglevel: log level to be used
        :param msg: message to be logged
        """
        loglevel = loglevel.upper()
        if loglevel == "DEBUG":
            self.logger.debug(msg)
        elif loglevel == "INFO":
            self.logger.info(msg)
        elif loglevel == "WARNING":
            self.logger.warning(msg)
        elif loglevel == "ERROR":
            self.logger.error(msg)
        elif loglevel == "CRITICAL":
            self.logger.critical(msg)

    def debug(self, msg):
        """Log debug message.

        :param msg: message to be logged
        """
        self.log("debug", msg)

    def info(self, msg):
        """Log info message.

        :param msg: message to be logged
        """
        self.log("info", msg)

    def warning(self, msg):
        """Log warning message.

        :param msg: message to be logged
        """
        self.log("warning", msg)

    def error(self, msg):
        """Log error message.

        :param msg: message to be logged
        """
        self.log("error", msg)

    def critical(self, msg):
        """Log critical message.

        :param msg: message to be logged
        """
        self.log("critical", msg)
