import torch


def create_optimizer(name: str, params, config):
    name = name.lower()

    if name == "sgd":
        return torch.optim.SGD(params, lr=config.learning_rate,
                                momentum=0.9,
                                weight_decay=config.weight_decay_rate)
    raise RuntimeError(f"Not found optimier {name}.")
