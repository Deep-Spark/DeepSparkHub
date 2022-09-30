import torch
import torch.nn as nn

from model.faster_rcnn import AnchorGenerator

from model import ops as ops
from model import faster_rcnn as faster_rcnn

from model.image_list import ImageList
from model.transform import GeneralizedRCNNTransform

from .pvanet import PVANetFeat


def get_pvanet_model_FRCNN(num_classes,
                            sizes=((32,), (64,), (128,), (256,), (512,)),
                            aspect_ratios=((0.5, 1.0, 2.0),)):
    '''   backbone is pvanet  '''
    backbone = PVANetFeat()
    backbone.out_channels = 512
    
    aspect_ratios_tmp = [(0.5, 1.0, 2.0) for i in range(len(sizes))]
    anchor_generator = AnchorGenerator(sizes=sizes, 
                                      aspect_ratios=aspect_ratios_tmp)
    roi_pooler = ops.MultiScaleRoIAlign(
                                            featmap_names=['0', '1', '2', '3'],
				            output_size=7,
					    sampling_ratio=2)
    model = faster_rcnn.FasterRCNN(backbone,
					num_classes,
					rpn_anchor_generator=anchor_generator,
					box_roi_pool=roi_pooler)
    return  model 


class BoxHead(nn.Module):
    def __init__(self, vgg):
        super(BoxHead, self).__init__()
        self.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])
        self.in_features = 4096 

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x


def generate_model(_faster_rcnn_cfg, **kwargs):
    model_name = _faster_rcnn_cfg["model_params"]["model_name"]
    assert model_name in list(model_func_dict.keys()),\
        "Model {} has not been supported in [{}].".format(model_name, 
            keys_str = "{} ".format(key for key in model_func_dict.keys())
        )
     
    func = model_func_dict[model_name]
    model = eval(func)(**kwargs)
    return model


def resize_boxes(boxes, original_size, new_size):
    # type: (Tensor, List[int], List[int]) -> Tensor
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


def postprocess(result,               # type: List[Dict[str, Tensor]]
                image_shapes,         # type: List[Tuple[int, int]]
                original_image_sizes  # type: List[Tuple[int, int]]
                ):
        # type: (...) -> List[Dict[str, Tensor]]
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
        return result


class RCNNTransformWrapper(nn.Module):
    def __init__(self):
        super(RCNNTransformWrapper, self).__init__()

    # List[Tensor*n]->ImageList
    def forward(self, images):
        image_sizes = [img.shape[-2:] for img in images]
        images = torch.stack(images)

        image_sizes_list: List[Tuple[int, int]] = []
        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)
        return image_list