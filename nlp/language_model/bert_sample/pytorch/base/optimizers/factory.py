# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.


from torch.optim import AdamW

from .lamb import Lamb


def create_optimizer(name: str, params, config):
    name = name.lower()

    if name == "lamb":
        return Lamb(
            params, lr=config.learning_rate,
            betas=(config.opt_lamb_beta_1, config.opt_lamb_beta_1), eps=1e-6,
            weight_decay=config.weight_decay_rate, adam=False
        )

    if name == "adamw":
        return AdamW(
            params, lr=config.learning_rate,
            betas=(config.opt_lamb_beta_1, config.opt_lamb_beta_2),
            weight_decay=config.weight_decay_rate
        )

    raise RuntimeError(f"Not found optimier {name}.")
