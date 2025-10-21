# Copyright (c) 2022, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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

import os
import sys
import re


os.chdir("./PaddleNLP")

# check infer log

expected = [1, 1, 1, 0, 0]
DELTA = 0.1
with open("./infer.log", "r") as f:
    log = f.read()
    labels = re.findall(r"Label\: (.*) ", log)
    Negative_probs = re.findall(r"Negative prob: (.*) ", log)
    Positive_probs = re.findall(r"Positive prob: (.*) ", log)
    print(labels)
    print(Negative_probs)
    print(Positive_probs)
    for i, (label, pos, neg) in enumerate(zip(labels, Positive_probs, Negative_probs)):
        delta = abs(float(expected[i]) - float(pos))
        print("Infer result: {}, {:.5f}, expected: {:.5f}".format(label, float(pos), float(expected[i])))
        if delta > DELTA:
            print("Check failed!")
            sys.exit(-1)
            