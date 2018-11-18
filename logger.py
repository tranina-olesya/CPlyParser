import logging


class CallCounted:
    def __init__(self, method):
        self.method = method
        self.counter = 0

    def __call__(self, *args, **kwargs):
        self.counter += 1
        return self.method(*args, **kwargs)


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.error = CallCounted(logger.error)