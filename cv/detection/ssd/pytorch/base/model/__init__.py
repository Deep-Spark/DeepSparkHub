import torch
from .models.ssd300 import SSD300


def load_checkpoint(model, checkpoint):
    print("loading model checkpoint", checkpoint)
    od = torch.load(checkpoint)

    # remove proceeding 'module' from checkpoint
    saved_model = od["model"]
    for k in list(saved_model.keys()):
        if k.startswith('module.'):
            saved_model[k[7:]] = saved_model.pop(k)
    model.load_state_dict(saved_model)
    return od


def create_model(config):
    model = SSD300(config)
    model.train()
    model.cuda()
    return model