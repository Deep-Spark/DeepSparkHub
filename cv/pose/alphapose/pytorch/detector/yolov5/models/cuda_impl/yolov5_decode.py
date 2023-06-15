import torch

try:
    from cuda_impl.yolov5_decode import yolov5_decode_forward
except:
    from ._ext.yolov5_decode import yolov5_decode_forward


class YoloV5DecodeForwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred_map, anchors, stride):
        return yolov5_decode_forward(pred_map, anchors, stride)


def yolov5_decode(pred_map, anchors, stride):
    num_anchors = len(anchors) // 2
    fm_shape = pred_map.shape
    W = fm_shape[-1]
    H = fm_shape[-2]
    B = fm_shape[0]
    if len(fm_shape) == 4:
        pred_map = pred_map.view(B, num_anchors, -1, H, W)
    # (B, num_anchors, 5+num_classses, H, W) -> (B, num_anchors, H, W, 5+num_classses)
    pred_map = pred_map.permute(0, 1, 3, 4, 2).contiguous()
    anchors = torch.FloatTensor(anchors).to(pred_map.device).type_as(pred_map)
    return YoloV5DecodeForwardFunction.apply(pred_map, anchors, stride)
