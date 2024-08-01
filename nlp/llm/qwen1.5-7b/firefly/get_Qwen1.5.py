#模型下载
from modelscope import snapshot_download
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, type=str, help="appoint the model to download, can be Qwen1.5-7B or Qwen1.5-14B")
    
    args = parser.parse_args()

    if args.model == "Qwen1.5-7B":
        model_dir = snapshot_download('qwen/Qwen1.5-7B')
    elif args.model == "Qwen1.5-14B":
        model_dir = snapshot_download('qwen/Qwen1.5-14B')

if __name__ == "__main__":
    main()