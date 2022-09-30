
import torch
import utils


class TrainingState:
    _trainer = None
    _status = 'aborted'  # later set to 'success' if termination criteria met

    iter_num = 0

    loss: float = 0.0
    avg_loss: float = 0.0
    base_lr: float = 0.0
    lr: float = 0.0

    epoch: int = 0
    end_training: bool = False
    converged: bool = False

    eval_ap = 0

    init_time = 0
    raw_train_time = 0

    def status(self):
        if self.converged:
            self._status = "success"
        return self._status

    def converged_success(self):
        self.end_training = True
        self.converged = True

    def to_dict(self, **kwargs):
        state_dict = dict()

        for var_name, value in self.__dict__.items():
            if not var_name.startswith("_") and utils.is_property(value):
                state_dict[var_name] = value

        exclude = ["eval_ap", "converged", "init_time", "raw_train_time"]
        for exkey in exclude:
            if exkey in state_dict:
                state_dict.pop(exkey)

        state_dict.update(kwargs)

        for k in state_dict.keys():
            if torch.is_tensor(state_dict[k]):
                state_dict[k] = state_dict[k].item()

        return state_dict
