from collections import OrderedDict
import json

from .logger import Logger
from .handlers import ConsoleHandler, JsonFileHandler

json_serializer = json.dumps

STR_FMT_DICT = dict(
    progress=' [{0: <20}]',
    metrics=' [{0: <28}]',
    default=' [{0: <10}]'
)


def get_str_format(key):
    if key not in STR_FMT_DICT:
        key = 'default'
    return STR_FMT_DICT[key]


class ConsoleFormatter:
    def __init__(self):
        pass

    def __call__(self, message):
        now = message['time']
        log_str = '{}-{}'.format(now.strftime('%y/%m/%d'), now.strftime('%H:%M:%S'))
        for key in message:
            if key == 'time':
                continue
            _msg = message[key]
            if isinstance(_msg, dict):
                _msg_list = [f'{k}: {v}' for k, v in _msg.items()]
                _log_str = ' '.join(_msg_list)
            else:
                _log_str = f'{key}: {_msg}'
            log_str += get_str_format(key).format(_log_str)
        log_str += '\n'
        return log_str


class JsonFormatter:
    def __init__(self):
        pass

    def __call__(self, message):
        _message = message.copy()
        now = _message['time']
        _message['time'] = OrderedDict(
            timestamp=now.timestamp(),
            date=now.strftime('%y/%m/%d'),
            time=now.strftime('%H:%M:%S')
        )
        log_str = "{}\n".format(json_serializer(_message))
        return log_str


class TrainingLogger(Logger):
    def __init__(self, log_name, handlers=[], **kwargs):
        handlers = [
            ConsoleHandler(formatter=ConsoleFormatter()),
            JsonFileHandler(
                formatter=JsonFormatter(), filepath=log_name, flush_freq=10)
        ]
        super().__init__(handlers=handlers)
