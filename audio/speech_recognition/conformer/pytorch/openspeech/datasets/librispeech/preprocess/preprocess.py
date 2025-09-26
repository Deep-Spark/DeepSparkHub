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


def collect_transcripts(
    dataset_path,
    librispeech_parts: list = [
        'train-clean-100',
        'train-clean-360',
        'train-other-500',
        'dev-clean',
        'dev-other',
        'test-clean',
        'test-other']
):
    """ Collect librispeech transcripts """
    transcripts_collection = list()

    for dataset in librispeech_parts:
        dataset_transcripts = list()

        for subfolder1 in os.listdir(os.path.join(dataset_path, dataset)):
            for subfolder2 in os.listdir(os.path.join(dataset_path, dataset, subfolder1)):
                for file in os.listdir(os.path.join(dataset_path, dataset, subfolder1, subfolder2)):
                    if file.endswith('txt'):
                        with open(os.path.join(dataset_path, dataset, subfolder1, subfolder2, file)) as f:
                            for line in f.readlines():
                                tokens = line.split()
                                audio_path = os.path.join(dataset, subfolder1, subfolder2, tokens[0])
                                audio_path = f"{audio_path}.flac"
                                transcript = " ".join(tokens[1:])
                                dataset_transcripts.append('%s|%s' % (audio_path, transcript))

                    else:
                        continue

        transcripts_collection.append(dataset_transcripts)

    return transcripts_collection
