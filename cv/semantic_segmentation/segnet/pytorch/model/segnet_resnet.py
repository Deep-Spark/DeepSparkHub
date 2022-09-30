# ref:

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

__all__ = ["SegNet"]

UPSAMPLE_MODE = "bilinear"
ALIGN_CORNERS = None


class SegNet(nn.Module):
    def __init__(self, classes=19):
        super(SegNet, self).__init__()

        backbone = resnet18(pretrained=True)

        batchNorm_momentum = 0.1

        self.stage1 = nn.Sequential(
            backbone.conv1, backbone.bn1
        )

        self.stage2 = nn.Sequential(
            backbone.maxpool,
            backbone.layer1
        )

        self.stage3 = backbone.layer2
        self.stage4 = backbone.layer3
        self.stage5 = backbone.layer4

        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv31d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv11d = nn.Conv2d(64, classes, kernel_size=3, padding=1)

        self.pooling = nn.MaxPool2d(2)

        self.aux0 = nn.Conv2d(64, classes, 1)
        self.aux1 = nn.Conv2d(128, classes, 1)

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)

        # Stage 5d
        # in: [B, 512, 7, 7], out: [B, 512, 14, 14]
        x5d = F.upsample(x5, size=x4.shape[2:], mode=UPSAMPLE_MODE, align_corners=ALIGN_CORNERS)
        x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = torch.cat([x4, x4], dim=1) + F.relu(self.bn51d(self.conv51d(x52d)))

        # Stage 4d
        # in: [B, 512, 14, 14], out: [B, 256, 28, 28]
        x4d = F.upsample(x51d, size=x3.shape[2:], mode=UPSAMPLE_MODE, align_corners=ALIGN_CORNERS)
        x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = torch.cat([x3, x3], dim=1) + F.relu(self.bn41d(self.conv41d(x42d)))

        # Stage 3d
        # in: [B, 256, 28, 28], out: [B, 128, 56, 56]
        x3d = F.upsample(x41d, size=x2.shape[2:], mode=UPSAMPLE_MODE, align_corners=ALIGN_CORNERS)
        x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = torch.cat([x2, x2], dim=1) + F.relu(self.bn31d(self.conv31d(x32d)))

        # Stage 2d
        # in: [B, 128, 56, 56], out: [B, 64, 112, 112]
        x2d = F.upsample(x31d, size=x1.shape[2:], mode=UPSAMPLE_MODE, align_corners=ALIGN_CORNERS)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = x1 + F.relu(self.bn21d(self.conv21d(x22d)))

        # Stage 1d
        # in: [B, 64, 112, 112], out: [B, nc, 224, 224]
        x1d = F.upsample(x21d, size=x.shape[2:], mode=UPSAMPLE_MODE, align_corners=ALIGN_CORNERS)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)

        if self.training:
            return dict(out=x11d, aux0=self.aux0(x21d), aux1=self.aux1(x31d))

        return x11d

    def load_from_segnet(self, model_path):
        s_dict = self.state_dict()  # create a copy of the state dict
        th = torch.load(model_path).state_dict()  # load the weigths
        # for name in th:
        # s_dict[corresp_name[name]] = th[name]
        self.load_state_dict(th)
