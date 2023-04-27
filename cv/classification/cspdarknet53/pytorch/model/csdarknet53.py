import torch.nn as nn
from .cslayers import *
import torch.backends.cudnn as cudnn
from torchsummary import summary

__all__ = ['CsDarkNet53']

class CsDarkNet53(nn.Module):
    def __init__(self, num_classes):
        super(CsDarkNet53, self).__init__()

        input_channels = 32

        # Network
        self.stage1 = Conv2dBatchLeaky(3, input_channels, 3, 1, activation='mish')
        self.stage2 = Stage2(input_channels)
        self.stage3 = Stage3(4*input_channels)
        self.stage4 = Stage(4*input_channels, 8)
        self.stage5 = Stage(8*input_channels, 8)
        self.stage6 = Stage(16*input_channels, 4)

        self.conv = Conv2dBatchLeaky(32*input_channels, 32*input_channels, 1, 1, activation='mish')
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        stage1 = self.stage1(x)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        stage5 = self.stage5(stage4)
        stage6 = self.stage6(stage5)

        conv = self.conv(stage6)
        x = self.avgpool(conv)
        x = x.view(-1, 1024)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
        cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    darknet = CsDarkNet53(num_classes=10)
    darknet = darknet.cuda()
    with torch.no_grad():
        darknet.eval()
        data = torch.rand(1, 3, 256, 256)
        data = data.cuda()
        try:
            #print(darknet)
            summary(darknet,(3,256,256))
            print(darknet(data))
        except Exception as e:
            print(e)
