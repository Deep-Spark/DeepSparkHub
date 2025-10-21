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

import os
import logging
from typing import Tuple, Optional

from openspeech.data.audio.dataset import SpeechToTextDataset
from openspeech.datasets import register_data_module
from openspeech.tokenizers.tokenizer import Tokenizer


@register_data_module('librispeech')
class LibriSpeechDataModule(object):
    """
    Data Module for LibriSpeech Dataset. LibriSpeech is a corpus of approximately 1000 hours of read
    English speech with sampling rate of 16 kHz, prepared by Vassil Panayotov with the assistance of Daniel Povey.
    The data is derived from read audiobooks from the LibriVox project, and has been carefully segmented and aligned.

    Args:
        configs (DictConfig): configuraion set
    """
    #LIBRISPEECH_TRAIN_NUM = 281241
    #LIBRISPEECH_VALID_NUM = 5567
    #LIBRISPEECH_TEST_NUM = 5559
    LIBRISPEECH_PARTS = [
        'dev-clean',
        'test-clean',
        'dev-other',
        'test-other',
        'train-clean-100',
        'train-clean-360',
        'train-other-500',
    ]

    def __init__(self, configs) -> None:
        super().__init__()
        self.configs = configs
        self.dataset = dict()
        self.logger = logging.getLogger(__name__)

    def _parse_manifest_file(self, manifest_file_path: str) -> Tuple[list, list]:
        """ Parsing manifest file """
        audio_paths = list()
        transcripts = list()

        with open(manifest_file_path, encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                audio_path, _, transcript = line.split('\t')
                transcript = transcript.replace('\n', '')

                audio_paths.append(audio_path)
                transcripts.append(transcript)

        return audio_paths, transcripts

    def prepare_data(self) -> Tokenizer:
        """
        Prepare librispeech data

        Returns:
            tokenizer (Tokenizer): tokenizer is in charge of preparing the inputs for a model.
        """
        if self.configs.tokenizer.unit == 'libri_subword':
            from openspeech.datasets.librispeech.preprocess.subword import generate_manifest_files
        elif self.configs.tokenizer.unit == 'libri_character':
            from openspeech.datasets.librispeech.preprocess.character import generate_manifest_files
        else:
            raise ValueError(f"Unsupported vocabulary unit: {self.configs.tokenizer.unit}")

        if self.configs.dataset.dataset_download:
            self._download_dataset()

        if not os.path.exists(self.configs.dataset.train_manifest_file):
            self.logger.info("Manifest file is not exists !!\n"
                             "Generate manifest files..")

            if hasattr(self.configs.tokenizer, "vocab_size"):
                generate_manifest_files(
                    dataset_path=self.configs.dataset.dataset_path,
                    manifest_file_path=self.configs.dataset.train_manifest_file,
                    vocab_path=self.configs.tokenizer.vocab_path,
                    vocab_size=self.configs.tokenizer.vocab_size,
                    librispeech_parts=self.configs.dataset.train_parts
                )
            else:
                generate_manifest_files(
                    dataset_path=self.configs.dataset.dataset_path,
                    manifest_file_path=self.configs.dataset.train_manifest_file,
                    vocab_path=self.configs.tokenizer.vocab_path,
                    librispeech_parts=self.configs.dataset.train_parts
                )

        if not os.path.exists(self.configs.dataset.eval_manifest_file):
            self.logger.info("Manifest file is not exists !!\n"
                             "Generate manifest files..")

            if hasattr(self.configs.tokenizer, "vocab_size"):
                generate_manifest_files(
                    dataset_path=self.configs.dataset.dataset_path,
                    manifest_file_path=self.configs.dataset.eval_manifest_file,
                    vocab_path=self.configs.tokenizer.vocab_path,
                    vocab_size=self.configs.tokenizer.vocab_size,
                    librispeech_parts=self.configs.dataset.eval_parts
                )
            else:
                generate_manifest_files(
                    dataset_path=self.configs.dataset.dataset_path,
                    manifest_file_path=self.configs.dataset.eval_manifest_file,
                    vocab_path=self.configs.tokenizer.vocab_path,
                    librispeech_parts=self.configs.dataset.eval_parts
                )

    def setup(
        self,
        stage: Optional[str] = None,
        tokenizer: Tokenizer = None,
        num_train_samples: int = None,
        num_eval_samples: int = None
    ) -> None:
        train_audio_paths, train_transcripts = self._parse_manifest_file(
            self.configs.dataset.train_manifest_file)
        eval_audio_paths, eval_transcripts = self._parse_manifest_file(
            self.configs.dataset.eval_manifest_file)

        if num_train_samples is None:
            num_train_samples = len(train_audio_paths)
        if num_eval_samples is None:
            num_eval_samples = len(eval_audio_paths)

        audio_paths = {
            "train": train_audio_paths[:num_train_samples],
            "val": eval_audio_paths[:num_eval_samples]
        }
        transcripts = {
            "train": train_transcripts[:num_train_samples],
            "val": eval_transcripts[:num_eval_samples]
        }

        for stage in audio_paths.keys():
            self.dataset[stage] = SpeechToTextDataset(
                configs=self.configs,
                dataset_path=self.configs.dataset.dataset_path,
                audio_paths=audio_paths[stage],
                transcripts=transcripts[stage],
                sos_id=tokenizer.sos_id,
                eos_id=tokenizer.eos_id,
                apply_spec_augment=self.configs.audio.apply_spec_augment if stage == 'train' else False,
                del_silence=self.configs.audio.del_silence if stage == 'train' else False,
            )
