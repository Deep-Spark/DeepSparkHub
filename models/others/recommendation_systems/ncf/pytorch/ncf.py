# Copyright (c) 2018, deepakn94, codyaustun, robieta. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# -----------------------------------------------------------------------
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.jit
from apex.optimizers import FusedAdam
import logging
import os
import sys
import math
import time
from datetime import datetime
from collections import OrderedDict
from argparse import ArgumentParser

import tqdm
import torch
import torch.nn as nn

import utils
from neumf import NeuMF

from mlperf_compliance import mlperf_log

from fp_optimizers import Fp16Optimizer
from apex.parallel import DistributedDataParallel as DDP

def parse_args():
    parser = ArgumentParser(description="Train a Nerual Collaborative"
                                        " Filtering model")
    parser.add_argument('data', type=str,
                        help='path to test and training data files')
    parser.add_argument('-e', '--epochs', type=int, default=20,
                        help='number of epochs for training')
    parser.add_argument('-b', '--batch-size', type=int, default=256,
                        help='number of examples for each iteration')
    parser.add_argument('--valid-batch-size', type=int, default=2**20,
                        help='number of examples in each validation chunk')
    parser.add_argument('-f', '--factors', type=int, default=8,
                        help='number of predictive factors')
    parser.add_argument('--layers', nargs='+', type=int,
                        default=[64, 32, 16, 8],
                        help='size of hidden layers for MLP')
    parser.add_argument('-n', '--negative-samples', type=int, default=4,
                        help='number of negative examples per interaction')
    parser.add_argument('-l', '--learning-rate', type=float, default=0.001,
                        help='learning rate for optimizer')
    parser.add_argument('-k', '--topk', type=int, default=10,
                        help='rank for test examples to be considered a hit')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use available GPUs')
    parser.add_argument('--seed', '-s', type=int,
                        help='manually set random seed for torch')
    parser.add_argument('--threshold', '-t', type=float,
                        help='stop training early at threshold')
    parser.add_argument('--fp16', action='store_true',
                        help='use fp16')
    parser.add_argument('--valid-negative', type=int, default=999,
                        help='Number of negative samples for each positive test example')
    parser.add_argument('--processes', '-p', type=int, default=1,
                        help='Number of processes for evaluating model')
    parser.add_argument('--workers', '-w', type=int, default=8,
                        help='Number of workers for training DataLoader')
    parser.add_argument('--beta1', '-b1', type=float,
                        help='beta1 for Adam')
    parser.add_argument('--beta2', '-b2', type=float,
                        help='beta1 for Adam')
    parser.add_argument('--eps', type=float,
                        help='eps for Adam')
    parser.add_argument('--loss-scale', default=8192, type=int,
                        help='Loss scale to use for fp16 training')
    parser.add_argument('--local_rank', default=0, type=int)
    return parser.parse_args()


def init_distributed(local_rank=0):
    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        '''
        Set cuda device so everything is done on the right GPU.
        THIS MUST BE DONE AS SOON AS POSSIBLE.
        '''
        torch.cuda.set_device(local_rank)

        '''Initialize distributed communication'''
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
    logger = logging.getLogger('mlperf_compliance')
    if local_rank > 0:
        sys.stdout = open('/dev/null', 'w')
        sys.stderr = open('/dev/null', 'w')
        logger.setLevel(logging.ERROR)

    return distributed, int(os.environ['WORLD_SIZE'])

def val_epoch(model, x, y, dup_mask, real_indices, K, samples_per_user, num_user, output=None,
              epoch=None, distributed=False):

    start = datetime.now()
    log_2 = math.log(2)

    model.eval()

    with torch.no_grad():
        p = []
        for u,n in zip(x,y):
            p.append(model(u, n, sigmoid=True).detach())

        del x
        del y
        temp = torch.cat(p).view(-1,samples_per_user)
        del p
        # set duplicate results for the same item to -1 before topk
        temp[dup_mask] = -1
        out = torch.topk(temp,K)[1]
        # topk in pytorch is stable(if not sort)
        # key(item):value(predicetion) pairs are ordered as original key(item) order
        # so we need the first position of real item(stored in real_indices) to check if it is in topk
        ifzero = (out == real_indices.view(-1,1))
        hits = ifzero.sum()
        ndcg = (log_2 / (torch.nonzero(ifzero)[:,1].view(-1).to(torch.float)+2).log_()).sum()

    mlperf_log.ncf_print(key=mlperf_log.EVAL_SIZE, value={"epoch": epoch, "value": num_user * samples_per_user})
    mlperf_log.ncf_print(key=mlperf_log.EVAL_HP_NUM_USERS, value=num_user)
    mlperf_log.ncf_print(key=mlperf_log.EVAL_HP_NUM_NEG, value=samples_per_user - 1)

    end = datetime.now()

    if distributed:
        torch.distributed.all_reduce(hits, op=torch.distributed.reduce_op.SUM)
        torch.distributed.all_reduce(ndcg, op=torch.distributed.reduce_op.SUM)

    hits = hits.item()
    ndcg = ndcg.item()

    if output is not None:
        result = OrderedDict()
        result['timestamp'] = datetime.now()
        result['duration'] = end - start
        result['epoch'] = epoch
        result['K'] = K
        result['hit_rate'] = hits/num_user
        result['NDCG'] = ndcg/num_user
        utils.save_result(result, output)

    return hits/num_user, ndcg/num_user

def generate_neg(users, true_mat, item_range, num_neg, sort=False):
    # assuming 1-d tensor input

    # for each user in 'users', generate 'num_neg' negative samples in [0, item_range)
    # also make sure negative sample is not in true sample set with mask
    # true_mat store a mask matrix where true_mat(user, item) = 0 for true sample
    # return (neg_user, neg_item)

    # list to append iterations of result
    neg_u = []
    neg_i = []

    neg_users = users.repeat(num_neg)
    while len(neg_users) > 0: # generate then filter loop
        neg_items = torch.empty_like(neg_users, dtype=torch.int64).random_(0, item_range)
        neg_mask = true_mat[neg_users, neg_items]
        neg_u.append(neg_users.masked_select(neg_mask))
        neg_i.append(neg_items.masked_select(neg_mask))

        neg_users = neg_users.masked_select(1-neg_mask)

    neg_users = torch.cat(neg_u)
    neg_items = torch.cat(neg_i)
    if sort == False:
        return neg_users, neg_items

    sorted_users, sort_indices = torch.sort(neg_users)
    return sorted_users, neg_items[sort_indices]

def main():

    args = parse_args()
    args.distributed, args.world_size = init_distributed(args.local_rank)
    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed)

    # Save configuration to file
    config = {k: v for k, v in args.__dict__.items()}
    config['timestamp'] = "{:.0f}".format(datetime.utcnow().timestamp())
    config['local_timestamp'] = str(datetime.now())
    run_dir = "./run/neumf/{}.{}".format(config['timestamp'],args.local_rank)
    print("Saving config and results to {}".format(run_dir))
    if not os.path.exists(run_dir) and run_dir != '':
        os.makedirs(run_dir)
    utils.save_config(config, run_dir)

    # Check that GPUs are actually available
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # more like load trigger timmer now
    mlperf_log.ncf_print(key=mlperf_log.PREPROC_HP_NUM_EVAL, value=args.valid_negative)
    # The default of np.random.choice is replace=True, so does pytorch random_()
    mlperf_log.ncf_print(key=mlperf_log.PREPROC_HP_SAMPLE_EVAL_REPLACEMENT, value=True)
    mlperf_log.ncf_print(key=mlperf_log.INPUT_HP_SAMPLE_TRAIN_REPLACEMENT)
    mlperf_log.ncf_print(key=mlperf_log.INPUT_STEP_EVAL_NEG_GEN)

    # sync worker before timing.
    if args.distributed:
        torch.distributed.broadcast(torch.tensor([1], device="cuda"), 0)
    torch.cuda.synchronize()

    #===========================================================================
    #== The clock starts on loading the preprocessed data. =====================
    #===========================================================================
    mlperf_log.ncf_print(key=mlperf_log.RUN_START)
    run_start_time = time.time()

    # load not converted data, just seperate one for test
    train_ratings = torch.load(args.data+'/train_ratings.pt', map_location=torch.device('cuda:{}'.format(args.local_rank)))
    test_ratings = torch.load(args.data+'/test_ratings.pt', map_location=torch.device('cuda:{}'.format(args.local_rank)))

    # get input data
    # get dims
    nb_maxs = torch.max(train_ratings, 0)[0]
    nb_users = nb_maxs[0].item()+1
    nb_items = nb_maxs[1].item()+1
    train_users = train_ratings[:,0]
    train_items = train_ratings[:,1]
    del nb_maxs, train_ratings
    mlperf_log.ncf_print(key=mlperf_log.INPUT_SIZE, value=len(train_users))
    # produce things not change between epoch
    # mask for filtering duplicates with real sample
    # note: test data is removed before create mask, same as reference
    mat = torch.cuda.ByteTensor(nb_users, nb_items).fill_(1)
    mat[train_users, train_items] = 0
    # create label
    train_label = torch.ones_like(train_users, dtype=torch.float32)
    neg_label = torch.zeros_like(train_label, dtype=torch.float32)
    neg_label = neg_label.repeat(args.negative_samples)
    train_label = torch.cat((train_label,neg_label))
    del neg_label
    if args.fp16:
        train_label = train_label.half()

    # produce validation negative sample on GPU
    all_test_users = test_ratings.shape[0]

    test_users = test_ratings[:,0]
    test_pos = test_ratings[:,1].reshape(-1,1)
    test_negs = generate_neg(test_users, mat, nb_items, args.valid_negative, True)[1]

    # create items with real sample at last position
    test_users = test_users.reshape(-1,1).repeat(1,1+args.valid_negative)
    test_items = torch.cat((test_negs.reshape(-1,args.valid_negative), test_pos), dim=1)
    del test_ratings, test_negs

    # generate dup mask and real indice for exact same behavior on duplication compare to reference
    # here we need a sort that is stable(keep order of duplicates)
    # this is a version works on integer
    sorted_items, indices = torch.sort(test_items) # [1,1,1,2], [3,1,0,2]
    sum_item_indices = sorted_items.float()+indices.float()/len(indices[0]) #[1.75,1.25,1.0,2.5]
    indices_order = torch.sort(sum_item_indices)[1] #[2,1,0,3]
    stable_indices = torch.gather(indices, 1, indices_order) #[0,1,3,2]
    # produce -1 mask
    dup_mask = (sorted_items[:,0:-1] == sorted_items[:,1:])
    dup_mask = torch.cat((torch.zeros_like(test_pos, dtype=torch.uint8), dup_mask),dim=1)
    dup_mask = torch.gather(dup_mask,1,stable_indices.sort()[1])
    # produce real sample indices to later check in topk
    sorted_items, indices = (test_items != test_pos).sort()
    sum_item_indices = sorted_items.float()+indices.float()/len(indices[0])
    indices_order = torch.sort(sum_item_indices)[1]
    stable_indices = torch.gather(indices, 1, indices_order)
    real_indices = stable_indices[:,0]
    del sorted_items, indices, sum_item_indices, indices_order, stable_indices, test_pos

    if args.distributed:
        test_users = torch.chunk(test_users, args.world_size)[args.local_rank]
        test_items = torch.chunk(test_items, args.world_size)[args.local_rank]
        dup_mask = torch.chunk(dup_mask, args.world_size)[args.local_rank]
        real_indices = torch.chunk(real_indices, args.world_size)[args.local_rank]

    # make pytorch memory behavior more consistent later
    torch.cuda.empty_cache()

    mlperf_log.ncf_print(key=mlperf_log.INPUT_BATCH_SIZE, value=args.batch_size)
    mlperf_log.ncf_print(key=mlperf_log.INPUT_ORDER)  # we shuffled later with randperm

    print('Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d'
          % (time.time()-run_start_time, nb_users, nb_items, len(train_users),
             nb_users))

    # Create model
    model = NeuMF(nb_users, nb_items,
                  mf_dim=args.factors, mf_reg=0.,
                  mlp_layer_sizes=args.layers,
                  mlp_layer_regs=[0. for i in args.layers])

    if args.fp16:
        model = model.half()

    print(model)
    print("{} parameters".format(utils.count_parameters(model)))

    # Save model text description
    with open(os.path.join(run_dir, 'model.txt'), 'w') as file:
        file.write(str(model))

    # Add optimizer and loss to graph
    if args.fp16:
        fp_optimizer = Fp16Optimizer(model, args.loss_scale)
        params = fp_optimizer.fp32_params
    else:
        params = model.parameters()

    #optimizer = torch.optim.Adam(params, lr=args.learning_rate, betas=(args.beta1, args.beta2), eps=args.eps)
    # optimizer = AdamOpt(params, lr=args.learning_rate, betas=(args.beta1, args.beta2), eps=args.eps)
    optimizer = FusedAdam(params, lr=args.learning_rate, betas=(args.beta1, args.beta2), eps=args.eps, eps_inside_sqrt=False)
    criterion = nn.BCEWithLogitsLoss(reduction = 'none') # use torch.mean() with dim later to avoid copy to host
    mlperf_log.ncf_print(key=mlperf_log.OPT_LR, value=args.learning_rate)
    mlperf_log.ncf_print(key=mlperf_log.OPT_NAME, value="Adam")
    mlperf_log.ncf_print(key=mlperf_log.OPT_HP_ADAM_BETA1, value=args.beta1)
    mlperf_log.ncf_print(key=mlperf_log.OPT_HP_ADAM_BETA2, value=args.beta2)
    mlperf_log.ncf_print(key=mlperf_log.OPT_HP_ADAM_EPSILON, value=args.eps)
    mlperf_log.ncf_print(key=mlperf_log.MODEL_HP_LOSS_FN, value=mlperf_log.BCE)

    if use_cuda:
        # Move model and loss to GPU
        model = model.cuda()
        criterion = criterion.cuda()

    if args.distributed:
        model = DDP(model)
        local_batch = args.batch_size // int(os.environ['WORLD_SIZE'])
    else:
        local_batch = args.batch_size
    traced_criterion = torch.jit.trace(criterion.forward, (torch.rand(local_batch,1),torch.rand(local_batch,1)))

    # Create files for tracking training
    valid_results_file = os.path.join(run_dir, 'valid_results.csv')
    # Calculate initial Hit Ratio and NDCG
    test_x = test_users.view(-1).split(args.valid_batch_size)
    test_y = test_items.view(-1).split(args.valid_batch_size)

    hr, ndcg = val_epoch(model, test_x, test_y, dup_mask, real_indices, args.topk, samples_per_user=test_items.size(1),
                         num_user=all_test_users, distributed=args.distributed)
    print('Initial HR@{K} = {hit_rate:.4f}, NDCG@{K} = {ndcg:.4f}'
          .format(K=args.topk, hit_rate=hr, ndcg=ndcg))
    success = False
    mlperf_log.ncf_print(key=mlperf_log.TRAIN_LOOP)
    for epoch in range(args.epochs):

        mlperf_log.ncf_print(key=mlperf_log.TRAIN_EPOCH, value=epoch)
        mlperf_log.ncf_print(key=mlperf_log.INPUT_HP_NUM_NEG, value=args.negative_samples)
        mlperf_log.ncf_print(key=mlperf_log.INPUT_STEP_TRAIN_NEG_GEN)

        begin = time.time()

        # prepare data for epoch
        neg_users, neg_items = generate_neg(train_users, mat, nb_items, args.negative_samples)
        epoch_users = torch.cat((train_users,neg_users))
        epoch_items = torch.cat((train_items,neg_items))
        del neg_users, neg_items

        # shuffle prepared data and split into batches
        epoch_indices = torch.randperm(len(epoch_users), device='cuda:{}'.format(args.local_rank))
        epoch_users = epoch_users[epoch_indices]
        epoch_items = epoch_items[epoch_indices]
        epoch_label = train_label[epoch_indices]
        if args.distributed:
            epoch_users = torch.chunk(epoch_users, args.world_size)[args.local_rank]
            epoch_items = torch.chunk(epoch_items, args.world_size)[args.local_rank]
            epoch_label = torch.chunk(epoch_label, args.world_size)[args.local_rank]
        epoch_users_list = epoch_users.split(local_batch)
        epoch_items_list = epoch_items.split(local_batch)
        epoch_label_list = epoch_label.split(local_batch)

        # only print progress bar on rank 0
        num_batches = (len(epoch_indices) + args.batch_size - 1) // args.batch_size
        if args.local_rank == 0:
            qbar = tqdm.tqdm(range(num_batches))
        else:
            qbar = range(num_batches)
        # handle extremely rare case where last batch size < number of worker
        if len(epoch_users_list) < num_batches:
            print("epoch_size % batch_size < number of worker!")
            exit(1)

        for i in qbar:
            # selecting input from prepared data
            user = epoch_users_list[i]
            item = epoch_items_list[i]
            label = epoch_label_list[i].view(-1,1)

            for p in model.parameters():
                p.grad = None

            outputs = model(user, item)
            loss = traced_criterion(outputs, label).float()
            loss = torch.mean(loss.view(-1), 0)

            if args.fp16:
                fp_optimizer.step(loss, optimizer)
            else:
                loss.backward()
                optimizer.step()

        del epoch_users, epoch_items, epoch_label, epoch_users_list, epoch_items_list, epoch_label_list, user, item, label
        train_time = time.time() - begin
        begin = time.time()

        mlperf_log.ncf_print(key=mlperf_log.EVAL_START)

        hr, ndcg = val_epoch(model, test_x, test_y, dup_mask, real_indices, args.topk, samples_per_user=test_items.size(1),
                             num_user=all_test_users, output=valid_results_file, epoch=epoch, distributed=args.distributed)

        val_time = time.time() - begin
        print('Epoch {epoch}: HR@{K} = {hit_rate:.4f}, NDCG@{K} = {ndcg:.4f},'
              ' train_time = {train_time:.2f}, val_time = {val_time:.2f}'
              .format(epoch=epoch, K=args.topk, hit_rate=hr,
                      ndcg=ndcg, train_time=train_time,
                      val_time=val_time))

        mlperf_log.ncf_print(key=mlperf_log.EVAL_ACCURACY, value={"epoch": epoch, "value": hr})
        mlperf_log.ncf_print(key=mlperf_log.EVAL_TARGET, value=args.threshold)
        mlperf_log.ncf_print(key=mlperf_log.EVAL_STOP)

        if args.threshold is not None:
            if hr >= args.threshold:
                print("Hit threshold of {}".format(args.threshold))
                success = True
                break

    mlperf_log.ncf_print(key=mlperf_log.RUN_STOP, value={"success": success})
    run_stop_time = time.time()
    mlperf_log.ncf_print(key=mlperf_log.RUN_FINAL)

    # easy way of tracking mlperf score
    if success:
        print("mlperf_score", run_stop_time - run_start_time)

if __name__ == '__main__':
    main()
