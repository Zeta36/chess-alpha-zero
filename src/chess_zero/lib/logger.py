"""
Logging helper methods
"""

from logging import StreamHandler, basicConfig, DEBUG, getLogger, Formatter


def setup_logger(log_filename):
    format_str = '%(asctime)s@%(name)s %(levelname)s # %(message)s'
    basicConfig(filename=log_filename, level=DEBUG, format=format_str)
    stream_handler = StreamHandler()
    stream_handler.setFormatter(Formatter(format_str))
    getLogger().addHandler(stream_handler)


if __name__ == '__main__':
    setup_logger("aa.log")
    logger = getLogger("test")
    logger.info("OK")
