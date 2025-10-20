from addict import Dict

class ConfigDict(Dict):

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        return super(ConfigDict, self).__getattr__(name)


from openspeech.tokenizers import TOKENIZER_REGISTRY
from openspeech.datasets import DATA_MODULE_REGISTRY


# PYTHONPATH=$PYTHONPATH:./ python test/test_dataloader.py
if __name__ == '__main__':
    config_file = 'configs/conformer_lstm.json'

    import json

    with open(config_file) as f:
        configs = json.load(f)

    configs = ConfigDict(configs)
    print(configs.model.model_name)

    data_module = DATA_MODULE_REGISTRY[configs.dataset.dataset](configs)
    data_module.prepare_data()
    tokenizer = TOKENIZER_REGISTRY[configs.tokenizer.unit](configs)

    data_module.setup(tokenizer=tokenizer)

    train_dataloader = data_module.train_dataloader()
    print(f'iters_per_epoch: {len(train_dataloader)}')
    steps = 0
    for batch_data in train_dataloader:
        inputs, targets, input_lengths, target_lengths = batch_data
        print(f'inputs: {inputs.size()} input_lengths: {input_lengths.size()} '
              f'targets: {targets.size()} target_lengths: {target_lengths.size()}')
        steps += 1
        if steps > 10:
            break
