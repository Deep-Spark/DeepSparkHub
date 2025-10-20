from addict import Dict

class ConfigDict(Dict):

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        return super(ConfigDict, self).__getattr__(name)


from openspeech.tokenizers import TOKENIZER_REGISTRY
from openspeech.models import MODEL_REGISTRY


# PYTHONPATH=$PYTHONPATH:./ python test/test_build_conformer.py
if __name__ == '__main__':
    config_file = 'configs/conformer_lstm.json'

    import json

    with open(config_file) as f:
        configs = json.load(f)

    configs = ConfigDict(configs)
    print(configs.model.model_name)

    tokenizer = TOKENIZER_REGISTRY[configs.tokenizer.unit](configs)

    model = MODEL_REGISTRY[configs.model.model_name](configs=configs, tokenizer=tokenizer)
    model.build_model()
    print("model:", model)
