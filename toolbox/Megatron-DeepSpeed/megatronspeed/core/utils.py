from megatron.training.global_vars import get_args
from megatron.core.utils import get_attr_wrapped_model

def get_model_config(model):
    args = get_args()
    if args.deepspeed and hasattr(model, 'module'):
        return get_attr_wrapped_model(model.module, 'config', allow_none=False)
    return get_attr_wrapped_model(model, 'config', allow_none=False)
