# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RTE dataset."""

from megatron_ds import print_rank_0
from tasks.data_utils import clean_text
from .data import GLUEAbstractDataset


LABELS = {'entailment': 0, 'not_entailment': 1}


class RTEDataset(GLUEAbstractDataset):

    def __init__(self, name, datapaths, tokenizer, max_seq_length,
                 test_label='entailment'):
        self.test_label = test_label
        super().__init__('RTE', name, datapaths,
                         tokenizer, max_seq_length)

    def process_samples_from_single_path(self, filename):
        """"Implement abstract method."""
        print_rank_0(' > Processing {} ...'.format(filename))

        samples = []
        total = 0
        first = True
        is_test = False
        with open(filename, 'r') as f:
            for line in f:
                row = line.strip().split('\t')
                if first:
                    first = False
                    if len(row) == 3:
                        is_test = True
                        print_rank_0('   reading {}, {}, and {} columns and '
                                     'setting labels to {}'.format(
                                         row[0].strip(), row[1].strip(),
                                         row[2].strip(), self.test_label))
                    else:
                        assert len(row) == 4
                        print_rank_0('    reading {}, {}, {}, and {} columns'
                                     ' ...'.format(
                                         row[0].strip(), row[1].strip(),
                                         row[2].strip(), row[3].strip()))
                    continue

                if is_test:
                    assert len(row) == 3, 'expected length 3: {}'.format(row)
                    uid = int(row[0].strip())
                    text_a = clean_text(row[1].strip())
                    text_b = clean_text(row[2].strip())
                    label = self.test_label
                    assert len(text_a) > 0
                    assert len(text_b) > 0
                else:
                    if len(row) == 4:
                        uid = int(row[0].strip())
                        text_a = clean_text(row[1].strip())
                        text_b = clean_text(row[2].strip())
                        label = row[-1].strip()
                    else:
                        print_rank_0('***WARNING*** index error, '
                                     'skipping: {}'.format(row))
                        continue
                    if len(text_a) == 0:
                        print_rank_0('***WARNING*** zero length a, '
                                     'skipping: {}'.format(row))
                        continue
                    if len(text_b) == 0:
                        print_rank_0('***WARNING*** zero length b, '
                                     'skipping: {}'.format(row))
                        continue
                assert label in LABELS
                assert uid >= 0

                sample = {'uid': uid,
                          'text_a': text_a,
                          'text_b': text_b,
                          'label': LABELS[label]}
                total += 1
                samples.append(sample)

                if total % 50000 == 0:
                    print_rank_0('  > processed {} so far ...'.format(total))

        print_rank_0(' >> processed {} samples.'.format(len(samples)))
        return samples
