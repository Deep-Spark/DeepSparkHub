import os
import sys
from math import ceil
# from mlperf_logging import mllog
# from mlperf_logging.mllog import constants

from model.unet3d import Unet3D
from model.losses import DiceCELoss, DiceScore

from data_loading.data_loader import get_data_loaders

from runtime.training import train
from runtime.inference import evaluate
from runtime.arguments import PARSER
from runtime.distributed_utils import init_distributed, get_world_size, get_device, is_main_process, get_rank
from runtime.distributed_utils import seed_everything, setup_seeds
# from runtime.logging import get_dllogger, mllog_start, mllog_end, mllog_event, mlperf_submission_log, mlperf_run_param_log
# from runtime.callbacks import get_callbacks
import warnings
warnings.filterwarnings("ignore")
DATASET_SIZE = 168
from ixpylogger import TrainingLogger

def main():
    flags = PARSER.parse_args()
    try:
        from dltest import show_training_arguments
        show_training_arguments(flags)
    except:
        pass
    
    logger = TrainingLogger(log_name=flags.log_name)
    
    local_rank = flags.local_rank
    device = get_device(local_rank)
    is_distributed = init_distributed()
    world_size = get_world_size()
    local_rank = get_rank()
    
    worker_seeds, shuffling_seeds = setup_seeds(flags.seed, flags.epochs, device)
    worker_seed = worker_seeds[local_rank]
    seed_everything(worker_seed)
    
    flags.seed = worker_seed
    model = Unet3D(1, 3, normalization=flags.normalization, activation=flags.activation)
    #print parameters total
    # from torchsummary import summary
    # print(model)
    # print(summary(model.cuda(),input_size=(1,128,128,128)))
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params:,} total parameters.')
    if flags.local_rank ==0:
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')
    #######################

    train_dataloader, val_dataloader = get_data_loaders(flags, num_shards=world_size, global_rank=local_rank)
    samples_per_epoch = world_size * len(train_dataloader) * flags.batch_size

    flags.evaluate_every = flags.evaluate_every or ceil(20*DATASET_SIZE/samples_per_epoch)
    flags.start_eval_at = flags.start_eval_at or ceil(1000*DATASET_SIZE/samples_per_epoch)

    
    loss_fn = DiceCELoss(to_onehot_y=True, use_softmax=True, layout=flags.layout,
                         include_background=flags.include_background)
    score_fn = DiceScore(to_onehot_y=True, use_argmax=True, layout=flags.layout,
                         include_background=flags.include_background)
    print(flags.exec_mode)

    if flags.exec_mode == 'train':
        is_successful = train(flags, model, train_dataloader, val_dataloader, loss_fn, score_fn,
                              device=device, is_distributed=is_distributed)
        if is_successful:
            sys.exit(0)
        else:
            sys.exit(1)
        

    elif flags.exec_mode == 'evaluate':
        eval_metrics = evaluate(flags, model, val_dataloader, loss_fn, score_fn,
                                device=device, is_distributed=is_distributed)
        if local_rank == 0:
            for key in eval_metrics.keys():
                print(key, eval_metrics[key])
    else:
        print("Invalid exec_mode.")
        pass


if __name__ == '__main__':
    main()
