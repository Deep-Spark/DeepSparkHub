# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
import torch
import argparse

from nerf.provider import NeRFDataset
from nerf.gui import NeRFGUI
from nerf.utils import *

from functools import partial
from loss import huber_loss

#torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if args.num_gpus > 1 and 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        env_rank = int(os.environ["RANK"])
        env_world_size = int(os.environ["WORLD_SIZE"])
        env_gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return

    dist_backend = "nccl"
    print('| distributed init (rank {}) (size {})'.format(env_rank,env_world_size), flush=True)
    torch.distributed.init_process_group(backend=dist_backend, init_method='env://',
                                         world_size=env_world_size, rank=env_rank)

    torch.cuda.set_device(env_gpu)
    torch.distributed.barrier()
    setup_for_distributed(env_rank == 0)


if __name__ == '__main__':
    print('[TIMESTAMP] start time:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--iters', type=int, default=30000, help="training iters")
    parser.add_argument('--epochs', type=int, default=-1, help="training epochs")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    ### test options
    parser.add_argument('--fps', type=int, default=15, help="test video fps")
    parser.add_argument('--duration', type=int, default=10, help="video duration second")
    parser.add_argument('--view', type=str, default='yaw', help="view direction:random or yaw")

    ### evaluate options
    parser.add_argument('--eval_interval', type=int, default=1, help="eval_interval")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")

    ### hash parameter
    parser.add_argument('--num_levels', type=int, default=16, help="hash number of levels")
    parser.add_argument('--log2_hashmap_size', type=int, default=19, help="hash map size of log2")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### experimental
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

    opt = parser.parse_args()

    init_distributed_mode(opt)

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True
    
    if opt.patch_size > 1:
        opt.error_map = False # do not use error_map if use patch-based training
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."


    if opt.ff:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --ff"
        from nerf.network_ff import NeRFNetwork
    elif opt.tcnn:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"
        from nerf.network_tcnn import NeRFNetwork
    else:
        from nerf.network import NeRFNetwork

    print(opt)
    
    seed_everything(opt.seed)

    model = NeRFNetwork(
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
        num_levels=opt.num_levels,
        log2_hashmap_size=opt.log2_hashmap_size,
    )
    
    print(model)

    criterion = torch.nn.MSELoss(reduction='none')
    #criterion = partial(huber_loss, reduction='none')
    #criterion = torch.nn.HuberLoss(reduction='none', beta=0.1) # only available after torch 1.10 ?

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.num_gpus > 1:
        env_rank = int(os.environ["RANK"])
        env_world_size = int(os.environ["WORLD_SIZE"])
        env_gpu = int(os.environ['LOCAL_RANK'])
    else:
        env_rank = 0
        env_world_size = 1
        env_gpu = 0

    if opt.test:
        
        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)

        if opt.gui:
            gui = NeRFGUI(opt, trainer)
            gui.render()
        
        else:
            test_loader = NeRFDataset(opt, device=device, type='test').dataloader()
            print('[TIMESTAMP] load data end:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

            if test_loader.has_gt:
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.
                print('[TIMESTAMP] evaluate end:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

            trainer.test(test_loader, write_video=True) # test and save video
            print('[TIMESTAMP] test end:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

            trainer.save_mesh(resolution=256, threshold=10)
            print('[TIMESTAMP] save mesh end:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    else:

        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

        train_loader = NeRFDataset(opt, device=device, type='train').dataloader()
        print('[TIMESTAMP] load data end:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        # decay to 0.1 * init_lr at last iter step
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        trainer = Trainer('ngp', opt, model, local_rank=env_gpu, world_size=env_world_size, device=device, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval)

        if opt.gui:
            gui = NeRFGUI(opt, trainer, train_loader)
            gui.render()
        
        else:
            valid_loader = NeRFDataset(opt, device=device, type='val', downscale=1).dataloader()

            max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
            if opt.epochs != -1:
                max_epoch = opt.epochs

            trainer.train(train_loader, valid_loader, max_epoch)
            print('[TIMESTAMP] trainning end:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

            # also test
            test_loader = NeRFDataset(opt, device=device, type='test').dataloader()
            
            if test_loader.has_gt:
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.
                print('[TIMESTAMP] evaluate end:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

            trainer.test(test_loader, write_video=True) # test and save video
            print('[TIMESTAMP] test end:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

            trainer.save_mesh(resolution=256, threshold=10)
            print('[TIMESTAMP] save mesh  end:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
