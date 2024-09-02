import json
import os
import argparse

def process(json_path, save_path, max_samples=None):
    parsed_data = []
    data = []
    with open(json_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError: {e.msg} in line {line}")
    
    for item in data:
        parsed_item = dict()
        parsed_item["conversations"] = [{"role": "user", "content": item["content"]}]
        parsed_item["conversations"].append({"role": "assistant", "content": item["summary"]})
        parsed_data.append(parsed_item)
        
    with open(save_path, 'w', encoding='utf-8') as outfile:
        for i, item in enumerate(parsed_data):
            if max_samples and i >= max_samples:
                print(f"note: save just {max_samples} max_samples to outfile")
                break
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write('\n')
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_max_samples', type=int, default=None, help='can get a small eval file if just for test function')
    args = parser.parse_args()

    train_json_path = "data/AdvertiseGen/train.json"
    process_train_json = "data/AdvertiseGen_process/train.json"
    eval_json_path = "data/AdvertiseGen/dev.json"
    process_eval_path = "data/AdvertiseGen_process/dev.json"
    if not os.path.exists(os.path.dirname(os.path.abspath(process_train_json))):
        os.mkdir(os.path.dirname(os.path.abspath(process_train_json)))
    
    print("process train datasets ...")
    process(train_json_path, process_train_json)

    print("process eval datasets ... ")
    process(eval_json_path, process_eval_path, max_samples=args.eval_max_samples)
