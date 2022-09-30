from cmath import log
from tqdm import tqdm
from collections import OrderedDict
import torch
from torch.optim import Adam, SGD
from torch.cuda.amp import autocast, GradScaler
import os
from runtime.distributed_utils import get_rank, reduce_tensor, get_world_size
from runtime.inference import evaluate
import time
from ixpylogger import TrainingLogger
import runtime.utils as utils

def get_optimizer(params, flags):
    if flags.optimizer == "adam":
        optim = Adam(params, lr=flags.learning_rate, weight_decay=flags.weight_decay)
    elif flags.optimizer == "sgd":
        optim = SGD(params, lr=flags.learning_rate, momentum=flags.momentum, nesterov=True,
                    weight_decay=flags.weight_decay)
    elif flags.optimizer == "lamb":
        import apex
        optim = apex.optimizers.FusedLAMB(params, lr=flags.learning_rate, betas=flags.lamb_betas,
                                          weight_decay=flags.weight_decay)
    else:
        raise ValueError("Optimizer {} unknown.".format(flags.optimizer))
    return optim


def lr_warmup(optimizer, init_lr, lr, current_epoch, warmup_epochs):
    scale = current_epoch / warmup_epochs
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr + (lr - init_lr) * scale


# def train(flags, model, train_loader, val_loader, loss_fn, score_fn, device, callbacks, is_distributed):
def train(flags, model, train_loader, val_loader, loss_fn, score_fn, device, is_distributed):
    logger = TrainingLogger(log_name=flags.log_name)
    rank = get_rank()
    world_size = get_world_size()
    torch.backends.cudnn.benchmark = flags.cudnn_benchmark
    torch.backends.cudnn.deterministic = flags.cudnn_deterministic

    optimizer = get_optimizer(model.parameters(), flags)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-3)

    scaler = GradScaler()
    
    model.to(device)
    loss_fn.to(device)

    model_without_ddp = model
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[flags.local_rank],
                                                          output_device=flags.local_rank)
        model_without_ddp = model.module
    
    if flags.resume:
        checkpoint = torch.load(flags.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        flags.start_epoch = checkpoint['epoch'] + 1

    metric_best = 0
    is_successful = False
    diverged = False
    next_eval_at = flags.start_eval_at
    if next_eval_at < flags.start_epoch:
        next_eval_at = flags.start_epoch -1 + flags.evaluate_every
    model.train()
    
    samples_per_epoch = world_size * len(train_loader) * flags.batch_size
    for epoch in range(flags.start_epoch, flags.epochs + 1):        
        model.train()
        cumulative_loss = []
        if epoch <= flags.lr_warmup_epochs and flags.lr_warmup_epochs > 0:
            lr_warmup(optimizer, flags.init_learning_rate, flags.learning_rate, epoch, flags.lr_warmup_epochs)
        else:
            scheduler.step()
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)
        
        model.train()
        loss_value = None
        optimizer.zero_grad()
        iter_time_start =time.time()
        for iteration, batch in enumerate(tqdm(train_loader, disable=(rank != 0) or not flags.verbose)):
            # torch.cuda.synchronize()
            # iter_time_start =time.time()
            image, label = batch
            image, label = image.to(device), label.to(device)
           
            
            with autocast(enabled=flags.amp):
                output = model(image)
                loss_value = loss_fn(output, label)
                loss_value /= flags.ga_steps

            if flags.amp:
                scaler.scale(loss_value).backward()
            else:
                loss_value.backward()

            if (iteration + 1) % flags.ga_steps == 0:
                if flags.amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()

            loss_value = reduce_tensor(loss_value, world_size).detach().cpu().numpy()
            
            cumulative_loss.append(loss_value)
            # torch.cuda.synchronize()
            # iter_time_end =time.time()
            if flags.local_rank ==0:
                logger.log(
                    OrderedDict(
                        progress=OrderedDict(
                            epoch=epoch,
                            step=iteration
                        ),
                        learning_rate=optimizer.param_groups[0]["lr"],
                        loss_value=loss_value.item()
                    )
                )
        torch.cuda.synchronize()
        iter_time_end =time.time()
        if flags.local_rank ==0:
                logger.log(
                    OrderedDict(
                        progress=OrderedDict(
                            fps=samples_per_epoch/(iter_time_end-iter_time_start)
                            
                        ),
                        
                    )
                )

        if epoch == next_eval_at:
            # print(f'epoch {epoch} eval_metrics ')
            next_eval_at += flags.evaluate_every
            del output
            
            eval_metrics = evaluate(flags, model, val_loader, loss_fn, score_fn, device, epoch)
            eval_metrics["train_loss"] = sum(cumulative_loss) / len(cumulative_loss)
            # print(f'eval_metrics  {eval_metrics}')
            if flags.local_rank ==0:
                logger.log(
                    OrderedDict(
                        progress=OrderedDict(
                            epoch=epoch,
                            step=iteration
                        ),
                        metrics=eval_metrics
                    )
                )
            
            if eval_metrics["mean_dice"] >= flags.quality_threshold:
                is_successful = True
            # elif eval_metrics["mean_dice"] < 1e-6:
            #     print("MODEL DIVERGED. ABORTING.")
                # diverged = True
            if flags.output_dir:
                checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'flags': flags}
                
                utils.save_on_master(
                    checkpoint,
                    os.path.join(flags.output_dir, 'model_{}.pth'.format(epoch)))
                
            if eval_metrics["mean_dice"] > metric_best:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(flags.output_dir, 'checkpoint.pth'))
                metric_best = eval_metrics["mean_dice"]

        if is_successful or diverged:
            break
        
    return is_successful
        

    
