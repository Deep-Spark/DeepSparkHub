# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json

with open('dataset/school_math/school_math_0.25M.jsonl', 'r', encoding='utf-8') as file:
    lines=file.readlines()

res_datas=[]
for line in lines:
    data=json.loads(line.strip())
    human_content=data["conversation"][0]["human"]
    assistant_content=data["conversation"][0]["assistant"]

    Res_data={"messages": [{"from": "human", "content": human_content}, {"from": "assistant", "content": assistant_content}]}

    res_datas.append(Res_data)
    # print(Res_data)
    if len(res_datas) > 10000:
        break

with open('dataset/school_math/convert/school_math_0.25M_convert.jsonl', 'w', encoding='utf-8') as file:
    for res_data in res_datas:
        file.write(json.dumps(res_data, ensure_ascii=False)+'\n')


