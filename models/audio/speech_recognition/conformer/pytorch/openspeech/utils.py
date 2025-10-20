# MIT License
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import torch
import platform
import importlib
from collections import OrderedDict
from typing import Tuple, Union, Iterable


PYTORCH_IMPORT_ERROR = """
Openspeech requires the PyTorch library but it was not found in your environment. Checkout the instructions on the
installation page: https://pytorch.org/get-started/locally/ and follow the ones that match your environment.
"""

TORCHAUDIO_IMPORT_ERROR = """
Openspeech requires the torchaudio library but it was not found in your environment. You can install it with pip:
`pip install torchaudio`
"""

LIBROSA_IMPORT_ERROR = """
Openspeech requires the librosa library but it was not found in your environment. You can install it with pip:
`pip install librosa`
"""

SENTENCEPIECE_IMPORT_ERROR = """
Openspeech requires the SentencePiece library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/google/sentencepiece#installation and follow the ones
that match your environment.
"""

WARPRNNT_IMPORT_ERROR = """
Openspeech requires the warp-rnnt library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/1ytic/warp-rnnt and follow the ones that match your environment.
"""

CTCDECODE_IMPORT_ERROR = """
Openspeech requires the ctcdecode library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/parlance/ctcdecode and follow the ones that match your environment.
"""

try:
    import librosa
except ImportError:
    raise ValueError(LIBROSA_IMPORT_ERROR)

DUMMY_SIGNALS, _ = librosa.load(librosa.ex('choice'))
DUMMY_FEATURES = librosa.feature.melspectrogram(y=DUMMY_SIGNALS, n_mels=80)
DUMMY_INPUTS = torch.FloatTensor(DUMMY_FEATURES).transpose(0, 1).unsqueeze(0).expand(3, -1, -1)
DUMMY_INPUT_LENGTHS = torch.IntTensor([1070, 900, 800])
DUMMY_TARGETS = torch.LongTensor([
    [2, 3, 3, 3, 3, 3, 2, 2, 1, 0],
    [2, 3, 3, 3, 3, 3, 2, 1, 2, 0],
    [2, 3, 3, 3, 3, 3, 2, 2, 0, 1],
])
DUMMY_TARGET_LENGTHS = torch.IntTensor([9, 8, 7])
DUMMY_TRANSCRIPTS = "OPENSPEECH IS AWESOME"

DUMMY_LM_INPUTS = torch.LongTensor([
    [2, 3, 3, 3, 3, 3, 2, 2, 0],
    [2, 3, 3, 3, 3, 3, 2, 3, 2],
    [2, 3, 3, 3, 3, 3, 2, 2, 0],
])
DYMMY_LM_INPUT_LENGTHS = torch.IntTensor([9, 8, 7])
DUMMY_LM_TARGETS = torch.LongTensor([
    [3, 3, 3, 3, 3, 2, 2, 1, 0],
    [3, 3, 3, 3, 3, 2, 1, 2, 0],
    [3, 3, 3, 3, 3, 2, 2, 0, 1],
])


def is_pytorch_available():
    return importlib.util.find_spec("torch") is not None


def is_librosa_available():
    return importlib.util.find_spec("librosa") is not None


def is_apex_available():
    return importlib.util.find_spec("apex") is not None


def is_sentencepiece_available():
    return importlib.util.find_spec("sentencepiece") is not None


def is_torchaudio_available():
    return importlib.util.find_spec("torchaudio") is not None


BACKENDS_MAPPING = OrderedDict(
    [
        ("torch", (is_pytorch_available, PYTORCH_IMPORT_ERROR)),
        ("sentencepiece", (is_sentencepiece_available, SENTENCEPIECE_IMPORT_ERROR)),
        ("librosa", (is_librosa_available, LIBROSA_IMPORT_ERROR)),
        ("torchaudio", (is_torchaudio_available, TORCHAUDIO_IMPORT_ERROR)),
    ]
)


def check_backends():
    backends = BACKENDS_MAPPING.keys()

    if not all(BACKENDS_MAPPING[backend][0]() for backend in backends):
        raise ImportError("".join([BACKENDS_MAPPING[backend][1] for backend in backends]))


def get_class_name(obj):
    return obj.__class__.__name__


def _check_environment(use_cuda: bool, logger) -> int:
    r"""
    Check execution envirionment.
    OS, Processor, CUDA version, Pytorch version, ... etc.
    """
    check_backends()

    cuda = use_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    logger.info(f"Operating System : {platform.system()} {platform.release()}")
    logger.info(f"Processor : {platform.processor()}")

    num_devices = torch.cuda.device_count()

    if str(device) == 'cuda':
        for idx in range(torch.cuda.device_count()):
            logger.info(f"device : {torch.cuda.get_device_name(idx)}")
        logger.info(f"CUDA is available : {torch.cuda.is_available()}")
        logger.info(f"CUDA version : {torch.version.cuda}")
        logger.info(f"PyTorch version : {torch.__version__}")

    else:
        logger.info(f"CUDA is available : {torch.cuda.is_available()}")
        logger.info(f"PyTorch version : {torch.__version__}")

    return num_devices


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def build_dummy_configs(
        model_configs=None,
        vocab_configs=None,
        criterion_configs=None,
        scheduler_configs=None,
        trainer_configs=None,
        audio_configs=None,
):
    from openspeech.models import ConformerConfigs
    from openspeech.criterion import CrossEntropyLossConfigs
    from openspeech.tokenizers.ksponspeech.character import KsponSpeechCharacterTokenizerConfigs
    from openspeech.data.audio.melspectrogram.melspectrogram import MelSpectrogramConfigs
    from openspeech.dataclass import GPUTrainerConfigs
    from openspeech.optim.scheduler.warmup_reduce_lr_on_plateau_scheduler import WarmupReduceLROnPlateauConfigs

    if model_configs is None:
        model_configs = ConformerConfigs()

    if vocab_configs is None:
        vocab_configs = KsponSpeechCharacterTokenizerConfigs()
        vocab_configs.vocab_path = "labels.csv"

    if criterion_configs is None:
        criterion_configs = CrossEntropyLossConfigs

    if trainer_configs is None:
        trainer_configs = GPUTrainerConfigs()

    if scheduler_configs is None:
        scheduler_configs = WarmupReduceLROnPlateauConfigs()

    if audio_configs is None:
        audio_configs = MelSpectrogramConfigs()

    return DotDict({
        'model': model_configs,
        'vocab': vocab_configs,
        'criterion': criterion_configs,
        'trainer': trainer_configs,
        'audio': audio_configs,
        'lr_scheduler': scheduler_configs,
    })

