from abc import ABC, abstractmethod
import atexit
from collections import deque
from datetime import datetime
from enum import IntEnum, unique


@unique
class Verbosity(IntEnum):
    NOTSET = 0
    DEBUG = 1
    INFO = 2
    WARNING = 3
    ERROR = 4
    CRITICAL = 5


class DummyFormatter:
    def __call__(self, message):
        return str(message)


class Handler(ABC):
    def __init__(
        self,
        verbosity=Verbosity.INFO,
        flush_freq=1,
        formatter=DummyFormatter(),
        filepath=None,
        append=False
    ):
        self._verbosity = verbosity
        self._flush_freq = flush_freq
        self._filepath = filepath
        if filepath is None:
            self._buffer = deque(maxlen=int(flush_freq * 2))
            self._cache_log = self._buffer_cache
        else:
            self._file = open(self._filepath, "a" if append else "w")
            atexit.register(self._file.close)
            self._cache_log = self._file_cache
        self._cache_size = 0
        self._formatter = formatter
        atexit.register(self.flush)

    @property
    def formatter(self):
        return self._formatter

    @formatter.setter
    def formatter(self, formatter):
        self._formatter = formatter

    @property
    def verbosity(self):
        return self._verbosity

    def _buffer_cache(self, format_message):
        self._buffer.append(format_message)

    def _file_cache(self, format_message):
        self._file.write(format_message)

    def log(self, message):
        self._cache_log(self.formatter(message))
        self._cache_size += 1
        if self._cache_size == self._flush_freq:
            self.flush()
            self._cache_size = 0

    @abstractmethod
    def flush(self):
        pass


class Logger:
    def __init__(self, handlers=[]):
        self.handlers = handlers

    def log(self, message, verbosity=Verbosity.INFO):
        now = datetime.now()
        message['time'] = now
        message.move_to_end('time', last=False)
        for handler in self.handlers:
            if handler.verbosity >= verbosity:
                handler.log(message)

    def add_handler(self, handler):
        self.handlers.append(handler)
