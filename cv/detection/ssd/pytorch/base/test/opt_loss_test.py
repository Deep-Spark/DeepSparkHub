import sys
sys.path.append("..")

import torch

from box_coder import dboxes300_coco
from model.losses.opt_loss import OptLoss
# from model.losses.loss import Loss

# In:
#  ploc : N x 8732 x 4
#  plabel : N x 8732
#  gloc : N x 8732 x 4
#  glabel : N x 8732

data = torch.load('loss.pth')
ploc = data['ploc'].cuda()
plabel = data['plabel'].cuda()
gloc = data['gloc'].cuda()
glabel = data['glabel'].cuda()

dboxes = dboxes300_coco()
# loss = Loss(dboxes).cuda()
opt_loss = OptLoss().cuda()

opt_loss = torch.jit.trace(opt_loss, (ploc, plabel, gloc, glabel))
# print(traced_loss.graph)

# timing
timing_iterations = 1000

import time

# Dry run to eliminate JIT compile overhead
dl = torch.tensor([1.], device="cuda")
l1 = opt_loss(ploc, plabel, gloc, glabel)
print("opt loss: {}".format(l1))
l1.backward()


# fprop
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    for _ in range(timing_iterations):
        l1 = opt_loss(ploc, plabel, gloc, glabel)

torch.cuda.synchronize()
end = time.time()

print('opt loss: {}'.format(l1))
time_per_fprop = (end - start) / timing_iterations

print('took {} seconds per iteration (fprop)'.format(time_per_fprop))

# fprop + bprop
torch.cuda.synchronize()
start = time.time()
for _ in range(timing_iterations):
    l1 = opt_loss(ploc, plabel, gloc, glabel)
    l1.backward()

torch.cuda.synchronize()
end = time.time()
print('opt loss: {}'.format(l1))

time_per_fprop_bprop = (end - start) / timing_iterations

print('took {} seconds per iteration (fprop + bprop)'.format(time_per_fprop_bprop))

# print(loss.graph_for(ploc, plabel, gloc, glabel))
