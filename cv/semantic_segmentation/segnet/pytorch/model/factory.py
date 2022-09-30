from . import segnet_bilinear, segnet_unpooling, segnet_bilinear_residual, segnet_resnet


def create_segnet(model_name="segnet_resnet", num_classes=11):
    return eval(model_name).SegNet(num_classes + 1)

