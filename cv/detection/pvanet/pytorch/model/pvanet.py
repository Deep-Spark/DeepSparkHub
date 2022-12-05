# Copyright (c) 2022, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
#!/usr/bin/env python
# coding=utf-8

"""
PVANet: Reference(https://arxiv.org/abs/1608.08021)
"""
import os
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F


class CReLU(nn.Module):
    def __init__(self, act=F.relu):
        super(CReLU, self).__init__()
        
        self.act = act 

    def forward(self, x):
        x = torch.cat((-x, x), dim=1)
        out = self.act(x)
        return out 


class ConvBnAct(nn.Module):
    def __init__(self, n_in, n_out, **kwargs):
        super(ConvBnAct, self).__init__()
        
        self.conv = nn.Conv2d(n_in, n_out, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(n_out)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class mCReLU_base(nn.Module):
    def __init__(self, n_in, n_out, kernelsize, stride=1, preAct=False, lastAct=True):
        super(mCReLU_base, self).__init__()

        # config 
        self._preAct = preAct 
        self._lastAct = lastAct
        self.act = nn.ReLU()

        # Trainable params 
        self.conv3x3 = nn.Conv2d(n_in, n_out, kernelsize, stride=stride, padding=int(kernelsize/2))
        self.bn = nn.BatchNorm2d(n_out * 2)
    
    def forward(self, x):
        if self._preAct:
            x = self.act(x)

        # Conv 3*3 - mCReLU (w / BN)
        x = self.conv3x3(x)
        x = torch.cat((-x, x), dim=1)
        x = self.bn(x)

        ##### TODO: Add scale-shift layer and make 'bn' optional #####
        if self._lastAct:
            x = self.act(x)
        
        return x


class mCReLU_residual(nn.Module):
    def __init__(self, n_in, n_red, n_3x3, n_out, kernelsize=3, in_stride=1, proj=False, preAct=False, lastAct=True):
        super(mCReLU_residual, self).__init__()
        self._preAct = preAct
        self._lastAct = lastAct
        self._stride = in_stride 
        self.act = nn.ReLU()

        # Trainable params 
        self.reduce = nn.Conv2d(n_in, n_red, 1, stride=in_stride)
        self.conv3x3 = nn.Conv2d(n_red, n_3x3, kernelsize, padding=int(kernelsize/2))
        self.bn = nn.BatchNorm2d(n_3x3 * 2)
        self.expand =  nn.Conv2d(n_3x3 * 2, n_out, 1)

        # move this assertion
        # if in_stride > 1:
        #     assert(proj)

        self.proj = nn.Conv2d(n_in, n_out, 1, stride=in_stride) if proj else None 
    
    def forward(self, x):
        x_sc = x

        if self._preAct:
            x = self.act(x)

        # Conv1x1 + ReLU--->reduce the channel
        x = self.reduce(x)
        x = self.act(x)

        # Conv3x3 + mCReLU (w / BN)
        x = self.conv3x3(x)
        x = torch.cat((-x, x), dim=1)
        x = self.bn(x)
        x = self.act(x)

        # Conv1x1 
        x = self.expand(x)

        if self._lastAct:
            x = self.act(x)

        # Projection
        if self.proj:
            x_sc = self.proj(x_sc)

        x = x + x_sc 
 
        return x 


class Inception(nn.Module):
    def __init__(self, n_in, n_out, in_stride=1, preAct=False, lastAct=True, proj=False):
        super(Inception, self).__init__()

        # config 
        self._preAct = preAct
        self._lastAct = lastAct
        self.n_in = n_in 
        self.n_out = n_out 
        self.in_stride = in_stride 
        self.act = nn.ReLU()   
        self.n_branches = 0
        self.n_outs = []      # number of output feature for each branch 

        self.proj = nn.Conv2d(n_in, n_out, 1, stride=in_stride) if proj else None

    def add_branch(self, module, n_out):
        # create branch
        br_name = 'branch_{}'.format(self.n_branches)
        setattr(self, br_name, module)
        
        # Last output chns 
        self.n_outs.append(n_out)
        self.n_branches += 1

    def branch(self, idx):
        br_name = 'branch_{}'.format(idx)
        return getattr(self, br_name, None)

    def add_convs(self, n_kernels, n_chns):
        assert (len(n_kernels) == len(n_chns))

        n_last = self.n_in
        layers = []

        stride = -1
        for k, n_out in zip(n_kernels, n_chns):
            if stride == -1:
                stride = self.in_stride
            else:
                stride = 1 

            # Initialize params 
            conv = nn.Conv2d(n_last, n_out, kernel_size=k, bias=False, padding=int(k / 2), stride=stride)
            bn = nn.BatchNorm2d(n_out)

            # Innstantiate network
            layers.append(conv)
            layers.append(bn)
            layers.append(self.act)

            n_last = n_out 
        
        self.add_branch(nn.Sequential(*layers), n_last)

    def add_poolconv(self, kernel, n_out, type='MAX'):
        assert (type in ['AVE', 'MAX'])
        n_last = self.n_in 
        layers = []

        # pooling 
        if type == 'MAX':
            layers.append(nn.MaxPool2d(kernel, padding=int(kernel / 2), stride=self.in_stride))
        elif type == 'AVE':
            layers.append(nn.AvgPool2d(kernel, padding=int(kernel / 2), stride=self.in_stride))

        # Conv-BN-Act 
        layers.append(nn.Conv2d(n_last, n_out, kernel_size=1))
        layers.append(nn.BatchNorm2d(n_out))
        layers.append(nn.ReLU())

        self.add_branch(nn.Sequential(*layers), n_out)

    def finalize(self):
        # add 1x1 conv 
        total_outs = sum(self.n_outs)

        self.last_conv = nn.Conv2d(total_outs, self.n_out, kernel_size=1)
        self.last_bn = nn.BatchNorm2d(self.n_out)
    
    def forward(self, x):
        x_sc = x

        if self._preAct:
            x = self.act(x)

        # compute branches 
        h = []
        for i in range(self.n_branches):
            module = self.branch(i)
            assert (module != None)

            h.append(module(x))

        x = torch.cat(h, dim=1)
        x = self.last_conv(x)
        x = self.last_bn(x)

        if self._lastAct:
            x = self.act(x)

        if (x_sc.get_device() != x.get_device()):
            print("Something's wrong!")

        # projection 
        if self.proj:
            x_sc = self.proj(x_sc)

        x = x + x_sc 
        return x
    

class PVANetFeat(nn.Module):
    def __init__(self):
        super(PVANetFeat, self).__init__()

        self.conv1 = nn.Sequential(
            mCReLU_base(3, 16, kernelsize=7, stride=2, lastAct=False),
            nn.MaxPool2d(3, padding=1, stride=2)
            )

        # 1/4
        self.conv2 = nn.Sequential(

            mCReLU_residual(32, 24, 24, 64, kernelsize=3, preAct=True, lastAct=False, in_stride=1, proj=True), 
            mCReLU_residual(64, 24, 24, 64, kernelsize=3, preAct=True, lastAct=False), 
            mCReLU_residual(64, 24, 24, 64, kernelsize=3, preAct=True, lastAct=False)
            )

        # 1/8
        self.conv3 = nn.Sequential(

            mCReLU_residual(64, 48, 48, 128, kernelsize=3, preAct=True, lastAct=False, in_stride=2, proj=True), 
            mCReLU_residual(128, 48, 48, 128, kernelsize=3, preAct=True, lastAct=False), 
            mCReLU_residual(128, 48, 48, 128, kernelsize=3, preAct=True, lastAct=False),
            mCReLU_residual(128, 48, 48, 128, kernelsize=3, preAct=True, lastAct=False)
            )

        # 1/16
        self.conv4 = nn.Sequential(
            self.gen_InceptionA(128, 2, True),
            self.gen_InceptionA(256, 1, False), 
            self.gen_InceptionA(256, 1, False), 
            self.gen_InceptionA(256, 1, False)
            )

        # 1/32
        self.conv5 = nn.Sequential(
            self.gen_InceptionB(256, 2, True),
            self.gen_InceptionB(384, 1, False), 
            self.gen_InceptionB(384, 1, False), 
            self.gen_InceptionB(384, 1, False), 

            nn.ReLU(inplace=True)
            )
    
        self.downscale = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.upscale = nn.ConvTranspose2d(in_channels=384, out_channels=384, kernel_size=4, stride=2, output_padding=0)
        self.upscale = nn.UpsamplingBilinear2d(scale_factor=2)
        self.convf = nn.Conv2d(in_channels=768, out_channels=512, kernel_size=1)
        
        self.f1 = nn.UpsamplingBilinear2d(scale_factor=4.)
        self.f2 = nn.UpsamplingBilinear2d(scale_factor=2.)

        self.f4 = nn.UpsamplingBilinear2d(scale_factor=0.5)
        self.f5 = nn.UpsamplingBilinear2d(scale_factor=0.25)


    def forward(self, x):           # x:  4*3*1056*640 
        x0 = self.conv1(x)          # x0: 4*32*264*160
        x1 = self.conv2(x0)         # x1: 4*64*264*160

        x2 = self.conv3(x1)         # x2:    4*128*132*80
        x2_ds = self.downscale(x2)  # x2_ds: 4*128*66*40

        x3 = self.conv4(x2)         # x3: 4*256*66*40 

        x4 = self.conv5(x3)         # x4:    4*384*33*20
        x4_us = self.upscale(x4)    # x4_us: 4*384*66*40
        
        out = torch.cat((x3, x2_ds, x4_us), dim=1)
        out = self.convf(out)

        f1 = self.f1(out)
        f2 = self.f2(out)
        f3 = out
        f4 = self.f4(out)
        f5 = self.f5(out)
        
        return {'0': f1, '1': f2, '2': f3, '3': f4, '4': f5}


    def gen_InceptionA(self, n_in, stride=1, poolconv=False, n_out=256):
        if (n_in != n_out) or (stride > 1):
            proj = True 
        else:
            proj = False

        module = Inception(n_in, n_out, preAct=True, lastAct=False, in_stride=stride, proj=proj)
        module.add_convs([1], [64])
        module.add_convs([1, 3], [48, 128])
        module.add_convs([1, 3, 3], [24, 48, 48])
        
        if poolconv:
            module.add_poolconv(3, 128)

        module.finalize()
        return module 
        
    def gen_InceptionB(self, n_in, stride=1, poolconv=False, n_out=384):
        if (n_in != n_out) or (stride > 1):
            proj = True 
        else:
            proj = False

        module = Inception(n_in, n_out, preAct=True, lastAct=False, in_stride=stride, proj=proj)
        module.add_convs([1], [64])
        module.add_convs([1, 3], [96, 192])
        module.add_convs([1, 3, 3], [32, 64, 64])
        
        if poolconv:
            module.add_poolconv(3, 128)

        module.finalize()
        return module 


class PVANet(nn.Module):
    def __init__(self, inputsize=(1068, 640), num_classes=1000):
        super(PVANet, self).__init__()

        # Follows torchvision naming convention
        self.features = PVANetFeat()

        assert (tmp % 16 == 0 for tmp in inputsize)
        fea_w = int(inputsize[0] / 16)
        fea_h = int(inputsize[1] / 16)

        self.classifier = nn.Sequential(
            nn.Linear(512 * fea_w * fea_h, 4096), 
                nn.BatchNorm1d(4096), 
                nn.ReLU(inplace=True), 
                nn.Dropout(), 

                nn.Linear(4096, 4096), 
                nn.BatchNorm1d(4096),
                nn.ReLU(inplace=True), 
                nn.Dropout(),

                nn.Linear(4096, num_classes)
            )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)     # reshape into (batchsize, all)
        x = self.classifier(x)

        return x 


def test_pvanetfeat():
    pvanetfeat = PVANetFeat()
    # inputx = torch.randn(4, 3, 1056, 640)        # 3*1056*640
    inputx = torch.randn(4, 3, 800, 800)           # 3*800*800
    pvanetfeat = pvanetfeat.cuda()
    inputx = inputx.cuda()
    
    outputx = pvanetfeat(inputx)                 # 512*66*40 

    for key, value in outputx.items():
        print(key, "===>", value.shape)


if __name__ == '__main__':
    print("Test PVANetFeat!")
    test_pvanetfeat()