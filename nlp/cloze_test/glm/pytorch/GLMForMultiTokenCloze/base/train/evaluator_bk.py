# coding=utf-8
import torch
import datetime
import subprocess

from utils import print_rank_0
from train.trainer import process_batch


class Evaluator:
    def __init__(self, args, dataloader):
        self.dataloader = dataloader
        self.args = args

    def evaluate(self, trainer):
        print_rank_0("calculating evaluate metrics ...")
        predictions, labels = multichoice_evaluate(
            trainer.model, self.dataloader, self.args)
        score = em_evaluate(predictions, labels)
        return score


def get_spare_port(args):
    if torch.distributed.get_rank() == 0:
        port = subprocess.check_output(
            ["shuf -n 1 -i 10000-65535"], shell=True)
        port = int(port.strip())
        # if port == args.master_port:
        #     port = subprocess.check_output(
        #         ["shuf -n 1 -i 10000-65535"], shell=True)
        #     port = int(port.strip())
        port = torch.cuda.LongTensor([port])
    else:
        port = torch.cuda.LongTensor([0])
    torch.distributed.broadcast(port, 0)
    port = port.item()
    return port


def multichoice_evaluate(model, dataloader, args, segment_length=10):
    model.eval()
    port = get_spare_port(args)
    print_rank_0(f"Using port {port}")
    store = torch.distributed.TCPStore("localhost", port,
                                       torch.distributed.get_world_size(),
                                       torch.distributed.get_rank() == 0, datetime.timedelta(seconds=30))
    total_sample = 0
    total_score = 0
    with torch.no_grad():
        # For all the batches in the dataset.
        for batch_index, batch in enumerate(dataloader):
            data = process_batch(batch,args.device)
            tokens, position_ids, attention_mask, target_ids, logit_mask = data[
                'text'], data['position'], data['mask'], data['target'], data['logit_mask']
            inputs = [tokens, position_ids,
                      attention_mask, target_ids, logit_mask]
            # if choice length max than 10
            if len(inputs[0].shape) == 3 and inputs[0].size(1) > segment_length:
                logit_list = []
                for i in range((inputs[0].size(1) - 1) // segment_length + 1):
                    input_batch = [
                        arg[:, i * segment_length: (i + 1) * segment_length] for arg in inputs]
                    logits, *mems = model(*input_batch)
                    logit_list.append(logits)
                logits = torch.cat(logit_list, dim=1)
            else:
                logits, *mems = model(*inputs)

            loss_mask = data["loss_mask"]
            logits = logits * loss_mask - 1000000000.0 * (1.0 - loss_mask)

            predicted = torch.argmax(logits, dim=-1).tolist()
            true_labels = data['answer_idx'].tolist()
            uids = batch['uid']

            for sample_id in range(len(true_labels)):
                uid = str(uids[sample_id])
                store.set(
                    uid, str((predicted[sample_id], true_labels[sample_id])))

    model.train()
    torch.distributed.barrier()
    predicted_labels = []
    true_labels = []
    for uid in range(len(dataloader.dataset)):
        uid = str(uid)
        prediction, true_choices = eval(store.get(uid))
        predicted_labels.append(prediction)
        true_labels.append(true_choices)
    torch.distributed.barrier()
    return predicted_labels, true_labels


def em_evaluate(predictions, labels):
    assert len(predictions) == len(labels)
    score = 0
    for pred, true_list in zip(predictions, labels):
        if pred in true_list:
            score += 1
    score = 100.0 * score / len(predictions)
    return score
