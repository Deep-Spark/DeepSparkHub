# 将原始数据转换成如下格式，保存成 jsonl 文件
# {"messages":
#   [
#     {
#       "from": "user",
#       "content": "what are some pranks with a pen i can do?"
#     },
#     {
#       "from": "assistant",
#       "content": "Are you looking for practical joke ideas?"
#     },
#     ...
#   ]
# },
# ...

import os
import json
from datasets import load_dataset

data_file = "../dataset/competition_math/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet"
data = load_dataset('parquet', data_files=data_file)["train"]

# 保存到 jsonl 文件中
save_dir = "../dataset/competition_math/sft"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'data.jsonl')

with open(save_path, 'w', encoding='utf-8') as fn:
    for line in data:
        user = {"from": "user", "content":line["problem"]}
        assistant = {"from": "assistant", "content":line["solution"]}
        info = {"messages":[user, assistant]}
        json.dump(info, fn)
        fn.write('\n')