# coding=utf-8
import torch
import os
import subprocess



def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def get_spare_port(args):
    if torch.distributed.get_rank() == 0:
        port = subprocess.check_output(["shuf -n 1 -i 10000-65535"], shell=True)
        port = int(port.strip())
        if port == args.master_port:
            port = subprocess.check_output(["shuf -n 1 -i 10000-65535"], shell=True)
            port = int(port.strip())
        port = torch.cuda.LongTensor([port])
    else:
        port = torch.cuda.LongTensor([0])
    torch.distributed.broadcast(port, 0)
    port = port.item()
    return port

def process_batch(batch):
    """Process batch and produce inputs for the model."""
    keys = list(batch.keys())
    if 'uid' in keys:
        keys.pop(keys.index('uid'))
    # Broadcast data.
    datatype = torch.int64
    data_b = mpu.broadcast_data(keys, batch, datatype)

    return data_b