from .logger import Handler


class ConsoleHandler(Handler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def flush(self):
        buf_size = len(self._buffer)
        for _ in range(buf_size):
            message = self._buffer.popleft()
            print(message, end='')


class JsonFileHandler(Handler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def flush(self):
        self._file.flush()
