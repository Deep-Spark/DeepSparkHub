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
    if len(res_datas) > 20000:
        break

with open('dataset/school_math/convert/school_math_0.25M_convert.jsonl', 'w', encoding='utf-8') as file:
    for res_data in res_datas:
        file.write(json.dumps(res_data, ensure_ascii=False)+'\n')


