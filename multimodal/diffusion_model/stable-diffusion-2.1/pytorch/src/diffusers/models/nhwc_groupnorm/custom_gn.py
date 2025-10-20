from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch, datetime, time, os, itertools
torch.set_printoptions(sci_mode=False)
module_dir = os.path.dirname(os.path.abspath(__file__))

from torch.utils.cpp_extension import load
gn_op = load(
        name="gn_op",
        sources=[
            os.path.join(module_dir, "custom_gn.cpp"),
            os.path.join(module_dir, "gn_kernel.cu"),
            #os.path.join(module_dir, "nchw_kernel.cu")
            ],
        extra_cuda_cflags=[
            '-use_fast_math',
            '-lineinfo', # useful for profiling
            ],
        extra_cflags=[
            '-O3', # needed or else GN NCHW from source is slower than nn.GroupNorm
            '-funroll-all-loops',
            '-march=native',
            ], 
        is_python_module=False,
        verbose=True,
        )

class GN_NHWC_Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, G: int, eps: float, activation: str):
        X_out, means, rstds = torch.ops.gnop.fwd(X, weight, bias, G, eps, activation)
        ctx.save_for_backward(X, weight, bias, means, rstds)
        ctx.G = G
        ctx.activation = activation
        return X_out

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        dy = dy.contiguous(memory_format=torch.channels_last)
        X, weight, bias, means, rstds = ctx.saved_tensors 
        dx, dgamma, dbeta = torch.ops.gnop.bwd(dy, X, weight, bias, means, rstds, ctx.G, ctx.activation)
        return dx, dgamma, dbeta, None, None, None

class GN_NHWC(nn.GroupNorm):
    def __init__(self, num_groups: int, nc: int, activation='identity', **kwargs):
        super().__init__(num_groups, nc, **kwargs)
        assert activation in {'identity', 'silu', 'relu', 'gelu', 'gelu_tanh'}
        if activation == 'identity':
            self.activation = 0
        if activation == 'relu':
            self.activation = 1
        if activation == 'silu':
            self.activation = 2
        if activation == 'gelu':
            self.activation = 3
        if activation == 'gelu_tanh':
            self.activation = 4

    @torch._dynamo.disable
    def forward(self, x):
        #print(x.shape, self.num_channels)
        if len(x.size()) == 3:
            N, C, L = x.shape
        elif len(x.size()) == 4:
            N, C, H, W = x.shape
        else:
            raise ValueError
        G = self.num_groups

        #if C // G > 512:
        #    raise ValueError(f'Error in fwd for X.shape={x.shape}, G={G}: C // G = {C // G} which is greater than 512. This input is not supported.')

        #if H * W % 8 != 0:
        #    raise ValueError(f'Error in fwd for X.shape={x.shape}, G={G}: H * W is not a multiple of 8. This input is not supported.')

        if self.affine:
            return GN_NHWC_Func.apply(x, self.weight, self.bias, self.num_groups, self.eps, self.activation)
        else:
            w = torch.ones((self.num_channels,), device=x.device, dtype=x.dtype)
            b = torch.zeros((self.num_channels,), device=x.device, dtype=x.dtype)
            return GN_NHWC_Func.apply(x, w, b, self.num_groups, self.eps, self.activation)

class GN_NCHW_Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, G: int, eps: float):
        X_out, means, rstds = gn_op.nchwforward(X, weight, bias, G, eps)
        ctx.save_for_backward(X, weight, means, rstds)
        ctx.G = G
        return X_out

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous()
        X, weight, means, rstds = ctx.saved_tensors 
        dx, dgamma, dbeta = gn_op.nchwbackward(dy, X, weight, means, rstds, ctx.G)
        return dx, dgamma, dbeta, None, None

class GN_NCHW(nn.GroupNorm):
    def __init__(self, num_groups: int, nc: int, **kwargs):
        super().__init__(num_groups, nc, **kwargs)

    def forward(self, x):
        if self.affine:
            return GN_NCHW_Func.apply(x.contiguous(), self.weight, self.bias, self.num_groups, self.eps)
        else:
            w = torch.ones((self.num_channels,), device=x.device, dtype=x.dtype)
            b = torch.zeros((self.num_channels,), device=x.device, dtype=x.dtype)
            return GN_NCHW_Func.apply(x.contiguous(), w, b, self.num_groups, self.eps)

def red(text):
    return '\033[91m' + str(text) + '\033[0m'
def green(text):
    return '\033[92m' + str(text) + '\033[0m'
def yellow(text):
    return '\033[93m' + str(text) + '\033[0m'
def blue(text):
    return '\033[94m' + str(text) + '\033[0m'

def config_filter(x): # returns true if config is valid
    DTYPE, B, C, R, G = x
    if C % G != 0:
        return False

    if R == 1: # this causes an autograd problem where it gets confused since the tensor is both contiguous in channels first/last format 
        return False

    if C / G > 512: # this isn't supported since it is assumed that at least one full group is processed per block in the fwd and the max threads per block is set to 512
        return False

    dtype_size = 2 if DTYPE in (torch.half, torch.bfloat16) else 4 # only care about 16/32-bit dtypes for now
    estimated_mem_usage_gib = (25 * dtype_size * B * C * R * R) / 2**30 #  this is just a rough estimate, likely wrong
    if estimated_mem_usage_gib > 4: # vram filter
        return False
    return True

if __name__ == '__main__':
    ACT_FN = 'silu'
    if ACT_FN == 'silu':
        act_fn = F.silu
    if ACT_FN == 'identity':
        act_fn = lambda x: x
    if ACT_FN == 'relu':
        act_fn = F.relu
    if ACT_FN == 'gelu':
        act_fn = F.gelu
    if ACT_FN == 'gelu_tanh':
        act_fn = lambda x: F.gelu(x, approximate='tanh')
    MODE = 'check' # can be 'check', 'bench', other modes do both
    CHECK_PROF = False

    if MODE != 'bench':
        #DTYPEs = (torch.bfloat16, torch.float, torch.double)
        DTYPEs = (torch.float16,)
        Bs = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 16)
        Cs = (
                32, 64, 128, 256, 512,
                13, 140, 125, 961,
                160, 320, 640, 960, 1280, 1600, 1920, 2240, 2560
                )
        Rs = (
                2, 3, 4, 5, 6, 7, 8, 9, 10, 17,
                8, 16, 64, 128, 256, 512,
                1024,
                )
        Gs = (1, 2, 4, 8, 16, 32,)
        all_params = itertools.product(DTYPEs, Bs, Cs, Rs, Gs)

        err_inputs = []
        for params in tqdm(sorted(
                #filter(config_filter, all_params),
                [
                    (torch.float16, 2, 640, 16, 32),
                    (torch.float16, 2, 1280, 8, 32),
                    (torch.float16, 2, 2560, 8, 32),
                    (torch.float16, 2, 1280, 16, 32),
                    (torch.float16, 2, 320, 32, 32),
                    (torch.float16, 2, 1920, 16, 32),
                    (torch.float16, 2, 2560, 16, 32),
                    (torch.float16, 2, 640, 32, 32),
                    (torch.float16, 2, 960, 32, 32),
                    (torch.float16, 2, 1280, 32, 32),
                    (torch.float16, 2, 320, 64, 32),
                    (torch.float16, 2, 1920, 32, 32),
                    (torch.float16, 2, 640, 64, 32),
                    (torch.float16, 2, 960, 64, 32),
                ],
                key = lambda x: x[1]*x[2]*x[3]*x[4]
        )):
            DTYPE, B, C, R, G = params
            #torch.cuda.empty_cache()
            print(f'B: {B:<2} | C: {C:<4} | R: {R:<4} | G: {G:<3} | DTYPE: {DTYPE}')
            x = torch.randn(B * C * R * R).reshape((B, C, R, R)).to(DTYPE, memory_format=torch.channels_last).cuda().requires_grad_(True) #* 1000
            #x = torch.arange(B * C * R * R).reshape((B, C, R, R)).to(DTYPE, memory_format=torch.channels_last).cuda().requires_grad_(True)/R/R/C/B-0.5 #* 1000
            torch.random.manual_seed(0)

            gn2 = GN_NHWC(G, C, activation=ACT_FN).cuda().to(DTYPE)

            if CHECK_PROF:
                #g1 = gn1(x.contiguous())
                #g1sum = g1.sum()
                #g1_grad_wrt_w = torch.autograd.grad(g1sum, gn1.weight, retain_graph=True)[0]
                g2 = gn2(x)
                #g2sum = g2.sum()
                #g2_grad_wrt_w = torch.autograd.grad(g2sum, gn2.weight, retain_graph=True)[0]
            else:
                gn1 = nn.GroupNorm(G, C).float().cuda()
                gn3 = nn.GroupNorm(G, C).cuda().to(DTYPE)
                with torch.no_grad():
                    w = torch.randn((C,), dtype=DTYPE)
                    b = torch.randn((C,), dtype=DTYPE)
                    gn1.weight.copy_(w.detach().float())
                    gn1.bias.copy_(b.detach().float())
                    gn2.weight.copy_(w.detach())
                    gn2.bias.copy_(b.detach())
                    gn3.weight.copy_(w.detach())
                    gn3.bias.copy_(b.detach())

                g1 = act_fn(gn1(x.float()))
                g2 = gn2(x)
                g3 = act_fn(gn3(x))
                rand_dy = torch.rand_like(g3)
                rand_dy /= rand_dy.numel() ** 0.5 # to prevent false positive errors from ocurring because of really large magnitude losses
                g1sum = (g1 * rand_dy).sum()
                g2sum = (g2 * rand_dy).sum()
                g3sum = (g3 * rand_dy).sum()
                def print_err(act_float, act_testing, act_ref, left_pad=0):
                    with torch.no_grad():
                        lpad = ' ' * left_pad
                        red_error = red('ERROR: ')
                        testing_err = F.mse_loss(act_float, act_testing)
                        expected_err = F.mse_loss(act_float, act_ref)
                        if testing_err.isnan() or testing_err / expected_err > 2 and testing_err > 1e-6:
                            print(red(f'{lpad}Your error: {testing_err}, expected error: {expected_err}'))
                            err_inputs.append((params, testing_err, expected_err))
                        else:
                            print(f'{lpad}Negligible difference (testing err: {testing_err:.2e}, ref err: {expected_err:.2e}) found')

                print('  FORWARD')
                print_err(g1, g2, g3, 4)
                print('  BACKWARD')
                print('    wrt weight')
                g1_grad_wrt_w = torch.autograd.grad(
                        g1sum, gn1.weight, retain_graph=True)[0]
                g2_grad_wrt_w = torch.autograd.grad(
                        g2sum, gn2.weight, retain_graph=True)[0]
                g3_grad_wrt_w = torch.autograd.grad(
                        g3sum, gn3.weight, retain_graph=True)[0]
                print_err(g1_grad_wrt_w, g2_grad_wrt_w, g3_grad_wrt_w, 6)


                print('    wrt bias')
                g1_grad_wrt_b = torch.autograd.grad(
                        g1sum, gn1.bias, retain_graph=True)[0]
                g2_grad_wrt_b = torch.autograd.grad(
                        g2sum, gn2.bias, retain_graph=True)[0]
                g3_grad_wrt_b = torch.autograd.grad(
                        g3sum, gn3.bias, retain_graph=True)[0]
                print_err(g1_grad_wrt_b, g2_grad_wrt_b, g3_grad_wrt_b, 6)

                print('    wrt X')
                g1_grad_wrt_x = torch.autograd.grad(g1sum, x, retain_graph=True)[0]
                g2_grad_wrt_x = torch.autograd.grad(g2sum, x, retain_graph=True)[0]
                g3_grad_wrt_x = torch.autograd.grad(g3sum, x, retain_graph=True)[0]
                print_err(g1_grad_wrt_x, g2_grad_wrt_x, g3_grad_wrt_x, 6)
        if len(err_inputs) > 0:
            print(red('Error inputs found:'))
            print('\n'.join(map(lambda x: f'{x[0]}, testing error: {x[1]:.2e}, expected error: {x[2]:.2e}', err_inputs)))
        elif not CHECK_PROF:
            print(green('No errors found :)'))

    if MODE != 'check':
        NSEC = 1 # number of seconds that each kernel runs for on a certain input
        DTYPES = [torch.bfloat16]
        BATCHES = [1, 2, 4, 8, 16, 32]
        #CHANNELS = [32, 64, 128, 256, 512]
        CHANNELS = [320, 640, 960, 1920, 2560]
        RESOLUTIONS = [4, 8, 16, 32, 64, 128, 256, 512]
        #NUM_GROUPS = [4, 8, 16, 32, 64, 128]
        NUM_GROUPS = [32]
        BENCH = 'fwd' # can be 'fwd', 'bwd', anything else is fwd + bwd
        GN_KERNELS = [
                #(GN_NHWC, 'GN NHWC fused (custom op)', gn_op.fwd_fused),
                #(GN_NHWC, 'GN NHWC NH grid (custom op)', gn_op.fwd_NH_grid),
                #(GN_NHWC, 'GN NHWC N grid (custom op)', gn_op.fwd_N_grid),
                #(GN_NHWC, 'GN NHWC NG grid NG grid (custom op)', gn_op.fwd_NG_grid),
                #(GN_NCHW, 'torch.nn GN NCHW (compiled from src)', gn_op.nchwforward),
                (nn.GroupNorm, 'torch.nn GN NCHW', None),
                #(GN_NCHW, 'torch.nn GN NCHW (compiled from src)', None),
                (GN_NHWC, 'GN NHWC', None),
        ]

        os.makedirs('csvs', exist_ok=True)
        fname = datetime.datetime.now().strftime("csvs/%H-%M-%S-%d-%m-%Y.csv")
        print(f'Writing to {fname}')
        outfile = open(fname, 'w')
        outfile.write('Kernel,B (batch),C (num channels),R (resolution),G (num groups), D (C/G),Speed (it/s; 25th percentile),Speed (it/s; 50th percentile),Speed (it/s; 75th percentile)\n')
        
        configs = list(filter(config_filter, itertools.product(DTYPES, BATCHES, CHANNELS, RESOLUTIONS, NUM_GROUPS)))
        print('Estimated time (seconds) to complete:', NSEC * len(configs) * len(GN_KERNELS))

        for DTYPE, B, C, R, G in configs:
            x_nchw = torch.randn((B, C, R, R), dtype=DTYPE, device='cuda').requires_grad_(True)
            x_nhwc = x_nchw.contiguous(memory_format=torch.channels_last).cuda().requires_grad_(True)

            gn_args = (G, C)
            print(BENCH, 'X shape:', x_nchw.shape, 'G (num groups):', G)
            for gn_class, desc, fwd_fn in GN_KERNELS:
                gn_input = x_nchw if 'NCHW' in desc else x_nhwc
                print(f'\t{desc}')

                try:
                    gn_layer = gn_class(*gn_args).cuda().to(DTYPE)
                    g = gn_layer(gn_input)
                    if not isinstance(gn_layer, GN_NHWC):
                        g = act_fn(g)

                    torch.cuda.synchronize()

                    tic = time.time()
                    tic_sec = time.time()
                    ntrials = 0
                    ntrials_minor = 0
                    minor_speeds = [] # used to track speed percentiles since they can often vary by a lot

                    while time.time() - tic < NSEC:
                        if BENCH == 'fwd':
                            if fwd_fn is None:
                                g = gn_layer(gn_input)
                            else:
                                g = fwd_fn(gn_input, gn_layer.weight, gn_layer.bias, gn_layer.num_groups, gn_layer.eps) # Not calling gn_layer(gn_input) since I found this added a lot of overhead
                        elif BENCH == 'both':
                            g = gn_layer(gn_input)
                        if not isinstance(gn_layer, GN_NHWC):
                            g = act_fn(g)
                        if BENCH != 'fwd':
                            torch.autograd.grad(g.sum(), gn_input, retain_graph=True)
                        torch.cuda.synchronize()

                        ntrials += 1
                        ntrials_minor += 1

                        if time.time() - tic_sec > 0.1:
                            speed = round(ntrials_minor / (time.time() - tic_sec), 2)
                            minor_speeds.append(speed)
                            print(f'\t\t{round(time.time() - tic, 1)}/{NSEC} seconds completed, speed: {blue(speed)} it/s\r', end='')
                            ntrials_minor = 0
                            tic_sec = time.time()

                    minor_speeds = np.array(minor_speeds)
                    median_speed = round(np.percentile(minor_speeds, 50), 2)
                    slow_speed = round(np.percentile(minor_speeds, 25), 2)
                    fast_speed = round(np.percentile(minor_speeds, 75), 2)
                    print(f'\n\t\tSpeed (25th/50th/75th percentile): {red(slow_speed)}/{yellow(median_speed)}/{green(fast_speed)} it/s')
                except KeyboardInterrupt:
                    print(f'Keyboard interrupt, closing {fname}.')
                    outfile.close()
                    raise
                except Exception as e:
                    print('\t\tFAILED; Error:', str(e).strip())
                    median_speed = slow_speed = fast_speed = '-1 (failed)'
                
                outfile.write(f'{desc},{B},{C},{R},{G},{C//G},{slow_speed},{median_speed},{fast_speed}\n')
            print()
        print(f'All tests done, closing {fname}.')
        outfile.close()
