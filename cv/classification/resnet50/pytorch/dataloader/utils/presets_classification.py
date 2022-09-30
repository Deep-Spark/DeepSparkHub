# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.
import torch
from torchvision.transforms import autoaugment, transforms


class ClassificationPresetTrain:
    def __init__(self, crop_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), hflip_prob=0.5,
                 auto_augment_policy=None, random_erase_prob=0.0, nhwc=False):
        trans = [transforms.RandomResizedCrop(crop_size)]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
            trans.append(autoaugment.AutoAugment(policy=aa_policy))
        trans.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))
        if nhwc:
            trans.append(add_nhwc())

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)

class add_nhwc:
    def __init__(self):
        pass
    def __call__(self, img):
        d1, d2, d3 = torch.split(img, 1, 0)
        d4 = torch.zeros(d1.size())
        return torch.cat([d1, d2, d3, d4], dim=0)
class ClassificationPresetEval:
    def __init__(self, crop_size, resize_size=256, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), nhwc=False):

        trans = [
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
        if nhwc:
            trans.append(add_nhwc())
        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)
