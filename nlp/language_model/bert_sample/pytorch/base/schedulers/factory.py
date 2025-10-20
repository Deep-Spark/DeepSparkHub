# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.


import config

from .linear_warmup_poly_scheduler import LinearWarmupPolyDecayScheduler
from .linear_warmup_scheduler import LinearWarmUpScheduler


def create_scheduler(optimizer, scheduler="poly"):
    if config.warmup_proportion == 0:
        warmup_steps = config.warmup_steps
        warmup_start = config.start_warmup_step
    else:
        warmup_steps = int(config.max_steps * config.warmup_proportion)
        warmup_start = 0

    if scheduler == "linear":
        return LinearWarmUpScheduler(optimizer, warmup_steps, config.max_steps)

    if scheduler == "poly":
        return LinearWarmupPolyDecayScheduler(optimizer, start_warmup_steps=warmup_start,
                                              warmup_steps=warmup_steps,
                                              total_steps=config.max_steps, end_learning_rate=0.0, degree=1.0)

    raise ValueError(f"Not found scheduler {scheduler}.")