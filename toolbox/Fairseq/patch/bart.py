# Copyright (c) 2023, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

from fairseq.models.bart import BARTModel

bart = BARTModel.from_pretrained(
    'checkpoints/bart',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='RTE-bin'
)

label_fn = lambda label: bart.task.label_dictionary.string(
    [label + bart.task.label_dictionary.nspecial]
)   
ncorrect, nsamples = 0, 0
bart.cuda()
bart.eval()
with open('glue_data/RTE/dev.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[1], tokens[2], tokens[3]
        tokens = bart.encode(sent1, sent2)
        prediction = bart.predict('sentence_classification_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        ncorrect += int(prediction_label == target)
        nsamples += 1
print('| Accuracy: ', float(ncorrect)/float(nsamples))