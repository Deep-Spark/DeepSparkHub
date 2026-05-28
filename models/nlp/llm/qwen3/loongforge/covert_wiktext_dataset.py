# Copyright (c) 2026, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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

from datasets import load_dataset
import json

ds = load_dataset("./dataset/wikitext/wikitext-103-raw-v1", split="train")
with open("./dataset/wikitext_train.jsonl", "w") as f:
    for item in ds:
        text = item["text"].strip()
        if text:
            f.write(json.dumps({"text": text}) + "\n")