"""Megatron global variables."""

import os
import sys
import torch

_GLOBAL_RLHF_ARGS = None

def get_rlhf_args():
    '''Return rlhf arguments.'''
    return _GLOBAL_RLHF_ARGS


def set_rlhf_args(rlhf_args):
    global _GLOBAL_RLHF_ARGS
    _GLOBAL_RLHF_ARGS = rlhf_args
