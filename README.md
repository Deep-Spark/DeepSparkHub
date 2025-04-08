[<img src="https://img.shields.io/badge/Language-English-blue.svg">](README_en.md) [<img src="https://img.shields.io/badge/语言-简体中文-red.svg">](README.md)

# DeepSparkHub

<div align="center" style="line-height: 1;">
  <a href="https://www.deepspark.org.cn"><img alt="Homepage"
    src="https://img.shields.io/badge/DeepSpark-Homepage-blue.svg"/></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-dfd.svg"></a>
  <a href="https://gitee.com/deep-spark/deepsparkhub/releases/latest"><img src="https://img.shields.io/github/v/release/deep-spark/deepsparkhub?color=ffa"></a>
</div>
<br>

DeepSparkHub甄选上百个应用算法和模型，覆盖AI和通用计算各领域，支持主流市场智能计算场景，包括智慧城市、数字个人、医疗、教育、通信、能源等多个领域。

## 模型库

### 大语言模型（LLM）

| Model                                                 | Framework | ToolBox            | Dataset/Weight         | IXUCA SDK |
|-------------------------------------------------------|-----------|--------------------|------------------------|-----------|
| [Aquila2-34B](nlp/llm/aquila2-34b/pytorch)            | PyTorch   | Megatron-DeepSpeed | Bookcorpus             | 3.4.0     |
| [Baichuan2-7B](nlp/llm/baichuan2-7b/pytorch)          | PyTorch   | DeepSpeed          | baichuan2-7b-base      | 3.4.0     |
| [Bloom-7B1](nlp/llm/bloom-7b1/pytorch)                | PyTorch   | Firefly            | school_math_0.25M      | 3.4.0     |
| [ChatGLM-6B](nlp/llm/chatglm-6b/pytorch)              | PyTorch   | DeepSpeed          | ADGEN & chatglm-6b     | 3.1.0     |
| [ChatGLM2-6B SFT](nlp/llm/chatglm2-6b-sft/pytorch)    | PyTorch   | DeepSpeed          | ADGEN & chatglm2-6b    | 3.4.0     |
| [ChatGLM3-6B](nlp/llm/chatglm3-6b/pytorch)            | PyTorch   | DeepSpeed          | ADGEN & chatglm3-6b    | 4.1.1     |
| [DeepSeekMoE 7B](nlp/llm/deepseek_moe_7b/pytorch)     | PyTorch   | ColossalAI         | deepseek-moe-16b-base  | 4.1.1     |
| [GLM-4](nlp/llm/glm-4/pytorch)                        | PyTorch   | Torchrun           | glm-4-9b-chat          | 4.2.0     |
| [Llama-7B](nlp/llm/llama-7b/pytorch)                  | PyTorch   | ColossalAI         | llama-7b-hf            | 3.1.0     |
| [Llama2-7B](nlp/llm/llama2-7b/pytorch)                | PyTorch   | Megatron-DeepSpeed | Bookcorpus             | 3.1.0     |
| [Llama2-7B RMF](nlp/llm/llama2-7b_reward_sft/pytorch) | PyTorch   | DeepSpeed          | Dahoas/rm-static       | 3.1.1     |
| [Llama2-7B RLHF](nlp/llm/llama2-7b_rlhf/pytorch)      | PyTorch   | Megatron-DeepSpeed | llama2-7b&tiny-llama   | 3.4.0     |
| [Llama2-7B SFT](nlp/llm/llama2-7b_sft/pytorch)        | PyTorch   | Megatron-DeepSpeed | GPT Small-117M         | 3.1.1     |
| [Llama2-13B](nlp/llm/llama2-13b/pytorch)              | PyTorch   | Megatron-DeepSpeed | Bookcorpus             | 3.4.0     |
| [Llama2-34B](nlp/llm/llama2-34b/pytorch)              | PyTorch   | Megatron-DeepSpeed | Bookcorpus             | 3.4.0     |
| [Llama3-8B](nlp/llm/llama3_8b/pytorch)                | PyTorch   | Megatron-DeepSpeed | Bookcorpus             | 4.1.1     |
| [Llama3-8B SFT](nlp/llm/llama3_8b_sft/pytorch)        | PyTorch   | ColossalAI         | school_math_0.25M      | 4.1.1     |
| [Mamba-2](nlp/llm/mamba-2/pytorch)                    | PyTorch   | Megatron-LM        | GPT Small-117M         | 4.1.1     |
| [MiniCPM](nlp/llm/minicpm/pytorch)                    | PyTorch   | DeepSpeed          | MiniCPM-2B-sft-bf16    | 4.2.0     |
| [Mixtral 8x7B](nlp/llm/mixtral/pytorch)               | PyTorch   | Megatron-LM        | GPT Small-117M         | 4.1.1     |
| [Phi-3](nlp/llm/phi-3/pytorch)                        | PyTorch   | Torchrun           | Phi-3-mini-4k-instruct | 4.2.0     |
| [QWen-7B](nlp/llm/qwen-7b/pytorch)                    | PyTorch   | Firefly            | qwen-7b                | 3.4.0     |
| [QWen1.5-7B](nlp/llm/qwen1.5-7b/pytorch)              | PyTorch   | Firefly            | school_math            | 4.1.1     |
| [QWen1.5-14B](nlp/llm/qwen1.5-14b/pytorch)            | PyTorch   | Firefly            | school_math            | 4.1.1     |
| [Qwen2.5-7B SFT](nlp/llm/qwen2.5-7b/pytorch)          | PyTorch   | LLaMA-Factory      | qwen2.5-7b             | 4.1.1     |
| [Yi-6B](nlp/llm/yi-6b/pytorch)                        | PyTorch   | DeepSpeed          | Yi-6B                  | 4.2.0     |
| [Yi-1.5-6B](nlp/llm/yi-1.5-6b/pytorch)                | PyTorch   | DeepSpeed          | Yi-1.5-6B              | 4.2.0     |
| [Yi-VL-6B](nlp/llm/yi-vl-6b/pytorch)                  | PyTorch   | LLaMA-Factory      | Yi-VL-6B-hf            | 4.2.0     |

### 计算机视觉（CV）

#### 视觉分类

| Model                                                                         | Framework    | Dataset  | IXUCA SDK |
|-------------------------------------------------------------------------------|--------------|----------|-------|
|  [ACmix](cv/classification/acmix/pytorch)                                       |  PyTorch       |  ImageNet  | 2.2.0 |
|  [ACNet](cv/classification/acnet/pytorch)                                       |  PyTorch       |  ImageNet  | 3.1.0 |
|  [AlexNet](cv/classification/alexnet/pytorch)                                   |  PyTorch       |  ImageNet  | 2.2.0 |
|  [AlexNet](cv/classification/alexnet/tensorflow)                                |  TensorFlow    |  ImageNet  | 3.1.0 |
|  [BYOL](cv/classification/byol/pytorch)                                         |  PyTorch       |  ImageNet  | 3.1.0 |
|  [CBAM](cv/classification/cbam/pytorch)                                         |  PyTorch       |  ImageNet  | 3.0.0 |
|  [ConvNext](cv/classification/convnext/pytorch)                                 |  PyTorch       |  ImageNet  | 2.2.0 |
|  [CspDarknet53](cv/classification/cspdarknet53/pytorch)                         |  PyTorch       |  ImageNet  | 3.0.0 |
|  [DenseNet](cv/classification/densenet/paddlepaddle)                            |  PaddlePaddle  |  ImageNet  | 3.1.0 |
|  [DenseNet](cv/classification/densenet/pytorch)                                 |  PyTorch       |  ImageNet  | 2.2.0 |
|  [DPN92](cv/classification/dpn92/pytorch)                                       |  PyTorch       |  ImageNet  | 2.2.0 |
|  [DPN107](cv/classification/dpn107/pytorch)                                     |  PyTorch       |  ImageNet  | 2.2.0 |
|  [ECA-MobileNetV2](cv/classification/eca_mobilenet_v2/pytorch)                  |  PyTorch       |  ImageNet  | 2.2.0 |
|  [ECA-ResNet152](cv/classification/eca_resnet152/pytorch)                       |  PyTorch       |  ImageNet  | 2.2.0 |
|  [EfficientNetB0](cv/classification/efficientnet_b0/paddlepaddle)               |  PaddlePaddle  |  ImageNet  | 3.1.0 |
|  [EfficientNetB4](cv/classification/efficientnet_b4/pytorch)                    |  PyTorch       |  ImageNet  | 2.2.0 |
|  [FasterNet](cv/classification/fasternet/pytorch)                               |  PyTorch       |  ImageNet  | 3.0.0 |
|  [GoogLeNet](cv/classification/googlenet/pytorch)                               |  PyTorch       |  ImageNet  | 2.2.0 |
|  [GoogLeNet](cv/classification/googlenet/paddlepaddle)                          |  PaddlePaddle  |  ImageNet  | 2.3.0 |
|  [InceptionV3](cv/classification/inceptionv3/mindspore)                         |  MindSpore     |  ImageNet  | 3.1.0 |
|  [InceptionV3](cv/classification/inceptionv3/pytorch)                           |  PyTorch       |  ImageNet  | 2.2.0 |
|  [InceptionV3](cv/classification/inceptionv3/tensorflow)                        |  TensorFlow    |  ImageNet  | 3.1.0 |
|  [InceptionV4](cv/classification/inceptionv4/pytorch)                           |  PyTorch       |  ImageNet  | 2.2.0 |
|  [InternImage](cv/classification/internimage/pytorch)                           |  PyTorch       |  ImageNet  | 3.1.0 |
|  [LeNet](cv/classification/lenet/pytorch)                                       |  PyTorch       |  ImageNet  | 2.2.0 |
|  [MobileNetV2](cv/classification/mobilenetv2/pytorch)                           |  PyTorch       |  ImageNet  | 2.2.0 |
|  [MobileNetV3](cv/classification/mobilenetv3/mindspore)                         |  MindSpore     |  ImageNet  | 3.1.0 |
|  [MobileNetV3](cv/classification/mobilenetv3/pytorch)                           |  PyTorch       |  ImageNet  | 2.2.0 |
|  [MobileNetV3](cv/classification/mobilenetv3/paddlepaddle)                      |  PaddlePaddle  |  ImageNet  | 2.3.0 |
|  [MobileNetV3_Large1.0](cv/classification/mobilenetv3_large_x1_0/paddlepaddle)  |  PaddlePaddle  |  ImageNet  | 3.1.0 |
|  [MobileOne](cv/classification/mobileone/pytorch)                               |  PyTorch       |  ImageNet  | 3.1.0 |
|  [MoCoV2](cv/classification/mocov2/pytorch)                                     |  PyTorch       |  ImageNet  | 3.1.0 |
|  [PP-LCNet](cv/classification/pp-lcnet/paddlepaddle)                            |  PaddlePaddle  |  ImageNet  | 3.1.0 |
|  [RepMLP](cv/classification/repmlp/pytorch)                                     |  PyTorch       |  ImageNet  | 3.1.0 |
|  [RepVGG](cv/classification/repvgg/pytorch)                                     |  PyTorch       |  ImageNet  | 3.0.0 |
|  [RepVGG](cv/classification/repvgg/paddlepaddle)                                |  PaddlePaddle  |  ImageNet  | 3.0.0 |
|  [RepViT](cv/classification/repvit/pytorch)                                     |  PyTorch       |  ImageNet  | 3.1.0 |
|  [Res2Net50_14w_8s](cv/classification/res2net50_14w_8s/paddlepaddle)            |  PaddlePaddle  |  ImageNet  | 3.1.0 |
|  [ResNeSt14](cv/classification/resnest14/pytorch)                               |  PyTorch       |  ImageNet  | 2.2.0 |
|  [ResNeSt50](cv/classification/resnest50/pytorch)                               |  PyTorch       |  ImageNet  | 2.2.0 |
|  [ResNeSt50](cv/classification/resnest50/paddlepaddle)                          |  PaddlePaddle  |  ImageNet  | 3.0.0 |
|  [ResNeSt101](cv/classification/resnest101/pytorch)                             |  PyTorch       |  ImageNet  | 2.2.0 |
|  [ResNeSt269](cv/classification/resnest269/pytorch)                             |  PyTorch       |  ImageNet  | 2.2.0 |
|  [ResNet18](cv/classification/resnet18/pytorch)                                 |  PyTorch       |  ImageNet  | 2.2.0 |
|  [ResNet50](cv/classification/resnet50/pytorch)                                 |  PyTorch       |  ImageNet  | 2.2.0 |
|  [ResNet50](cv/classification/resnet50/paddlepaddle)                            |  PaddlePaddle  |  ImageNet  | 2.3.0 |
|  [ResNet50](cv/classification/resnet50/tensorflow)                              |  TensorFlow    |  ImageNet  | 3.0.0 |
|  [ResNet101](cv/classification/resnet101/pytorch)                               |  PyTorch       |  ImageNet  | 2.2.0 |
|  [ResNet152](cv/classification/resnet152/pytorch)                               |  PyTorch       |  ImageNet  | 2.2.0 |
|  [ResNeXt50_32x4d](cv/classification/resnext50_32x4d/mindspore)                 |  MindSpore     |  ImageNet  | 3.1.0 |
|  [ResNeXt50_32x4d](cv/classification/resnext50_32x4d/pytorch)                   |  PyTorch       |  ImageNet  | 2.2.0 |
|  [ResNeXt101_32x8d](cv/classification/resnext101_32x8d/pytorch)                 |  PyTorch       |  ImageNet  | 2.2.0 |
|  [SE_ResNet50_vd](cv/classification/se_resnet50_vd/paddlepaddle)                |  PaddlePaddle  |  ImageNet  | 3.1.0 |
|  [SEResNeXt](cv/classification/seresnext/pytorch)                               |  PyTorch       |  ImageNet  | 2.2.0 |
|  [ShuffleNetV2](cv/classification/shufflenetv2/paddlepaddle)                    |  PaddlePaddle  |  ImageNet  | 3.1.0 |
|  [ShuffleNetV2](cv/classification/shufflenetv2/pytorch)                         |  PyTorch       |  ImageNet  | 2.2.0 |
|  [SqueezeNet](cv/classification/squeezenet/pytorch)                             |  PyTorch       |  ImageNet  | 2.2.0 |
|  [Swin Transformer](cv/classification/swin_transformer/paddlepaddle)            |  PaddlePaddle  |  ImageNet  | 3.0.0 |
|  [Swin Transformer](cv/classification/swin_transformer/pytorch)                 |  PyTorch       |  ImageNet  | 2.2.0 |
|  [VGG16](cv/classification/vgg/paddlepaddle)                                    |  PaddlePaddle  |  ImageNet  | 2.3.0 |
|  [VGG16](cv/classification/vgg/pytorch)                                         |  PyTorch       |  ImageNet  | 2.2.0 |
|  [VGG16](cv/classification/vgg/tensorflow)                                      |  TensorFlow    |  ImageNet  | 3.1.0 |
|  [Wave-MLP](cv/classification/wavemlp/pytorch)                                  |  PyTorch       |  ImageNet  | 2.2.0 |
|  [Wide_ResNet101_2](cv/classification/wide_resnet101_2/pytorch)                 |  PyTorch       |  ImageNet  | 2.2.0 |
|  [Xception](cv/classification/xception/paddlepaddle)                            |  PaddlePaddle  |  ImageNet  | 3.1.0 |
|  [Xception](cv/classification/xception/pytorch)                                 |  PyTorch       |  ImageNet  | 2.2.0 |

#### 人脸检测

| Model                                              | Framework | Dataset    | IXUCA SDK |
|----------------------------------------------------|-----------|------------|-------|
|  [RetinaFace](cv/face_detection/retinaface/pytorch)  |  PyTorch    |  WIDER FACE  | 3.0.0 |

#### 人脸识别

| Model                                                   | Framework    | Dataset            | IXUCA SDK |
|---------------------------------------------------------|--------------|--------------------|-------|
|  [ArcFace](cv/face_recognition/arcface/pytorch)          |  PyTorch       |  CASIA-WebFaces&LFW  | 3.0.0 |
|  [BlazeFace](cv/face_recognition/blazeface/paddlepaddle)  |  PaddlePaddle  |  WIDER FACE          | 3.1.0 |
|  [CosFace](cv/face_recognition/cosface/pytorch)           |  PyTorch       |  CASIA-WebFaces&LFW  | 3.0.0 |
|  [FaceNet](cv/face_recognition/facenet/pytorch)           |  PyTorch       |  CASIA-WebFaces&LFW  | 3.0.0 |
|  [FaceNet](cv/face_recognition/facenet/tensorflow)        |  TensorFlow    |  CASIA-WebFaces&LFW  | 3.1.0 |

#### 实例分割

| Model                                               | Framework    | Dataset | IXUCA SDK |
|-----------------------------------------------------|--------------|---------|-------|
|  [SOLO](cv/instance_segmentation/solo/pytorch)        |  PyTorch       |  COCO     | 3.0.0 |
|  [SOLOv2](cv/detection/solov2/paddlepaddle)           |  PaddlePaddle  |  COCO     | 3.0.0 |
|  [SOLOv2](cv/instance_segmentation/solov2/pytorch)    |  PyTorch       |  COCO     | 3.1.0 |
|  [YOLACT++](cv/instance_segmentation/yolact/pytorch)  |  PyTorch       |  COCO     | 3.0.0 |

#### 图像生成

| Model                                               | Framework    | Dataset  | IXUCA SDK |
|-----------------------------------------------------|--------------|----------|-------|
|  [DCGAN](cv/image_generation/dcgan/mindspore)         |  MindSpore     |  ImageNet  | 3.0.0 |
|  [Pix2Pix](cv/image_generation/pix2pix/paddlepaddle)  |  PaddlePaddle  |  facades   | 3.1.0 |

#### 知识蒸馏

| Model                             | Framework | Dataset      | IXUCA SDK |
|-----------------------------------|-----------|--------------|-------|
|  [CWD](cv/distiller/cwd/pytorch)    |  PyTorch    |  Cityscapes    | 3.0.0 |
|  [RKD](cv/distiller/rkd/pytorch)    |  PyTorch    |  CUB-200-2011  | 3.0.0 |
|  [WSLD](cv/distiller/wsld/pytorch)  |  PyTorch    |  ImageNet      | 3.1.0 |

#### 目标检测

| Model                                                         | Framework             | Dataset | IXUCA SDK |
|---------------------------------------------------------------|-----------------------|---------|-------|
|  [ATSS](cv/detection/atss_mmdet/pytorch)                        |  PyTorch (MMDetection)  |  COCO     | 3.0.0 |
|  [AutoAssign](cv/detection/autoassign/pytorch)                  |  PyTorch                |  COCO     | 2.2.0 |
|  [Cascade R-CNN](cv/detection/cascade_rcnn_mmdet/pytorch)       |  PyTorch (MMDetection)  |  COCO     | 3.0.0 |
|  [CenterMask2](cv/detection/centermask2/pytorch)                |  PyTorch                |  COCO     | 4.1.1 |
|  [CenterNet](cv/detection/centernet/pytorch)                    |  PyTorch                |  COCO     | 2.2.0 |
|  [CenterNet](cv/detection/centernet/paddlepaddle)               |  PaddlePaddle           |  COCO     | 3.0.0 |
|  [Co-DETR](cv/detection/co-detr/pytorch)                        |  PyTorch                |  COCO     | 3.1.0 |
|  [CornerNet](cv/detection/cornernet_mmdet/pytorch)              |  PyTorch (MMDetection)  |  COCO     | 3.0.0 |
|  [DCNV2](cv/detection/dcnv2_mmdet/pytorch)                      |  PyTorch (MMDetection)  |  COCO     | 3.0.0 |
|  [DETR](cv/detection/detr/paddlepaddle)                         |  PaddlePaddle           |  COCO     | 3.0.0 |
|  [Faster R-CNN](cv/detection/fasterrcnn/pytorch)                |  PyTorch                |  COCO     | 2.2.0 |
|  [FCOS](cv/detection/fcos/paddlepaddle)                         |  PaddlePaddle           |  COCO     | 3.0.0 |
|  [FCOS](cv/detection/fcos/pytorch)                              |  PyTorch                |  COCO     | 3.0.0 |
|  [Mamba-YOLO](cv/detection/mamba_yolo/pytorch)                  |  PyTorch                |  COCO     | 4.1.1 |
|  [Mask R-CNN](cv/detection/maskrcnn/pytorch)                    |  PyTorch                |  COCO     | 2.2.0 |
|  [Mask R-CNN](cv/detection/maskrcnn/paddlepaddle)               |  PaddlePaddle           |  COCO     | 2.3.0 |
|  [OC_SORT](cv/detection/oc_sort/paddlepaddle)                   |  PaddlePaddle           |  MOT17    | 3.1.0 |
|  [Oriented RepPoints](cv/detection/oriented_reppoints/pytorch)  |  PyTorch                |  DOTA     | 3.1.0 |
|  [PP-PicoDet](cv/detection/picodet/paddlepaddle)                |  PaddlePaddle           |  COCO     | 3.1.0 |
|  [PP-YOLOE](cv/detection/pp-yoloe/paddlepaddle)                 |  PaddlePaddle           |  COCO     | 2.3.0 |
|  [PP-YOLOE+](cv/detection/pp_yoloe+/paddlepaddle)               |  PaddlePaddle           |  COCO     | 3.1.1 |
|  [PVANet](cv/detection/pvanet/pytorch)                          |  PyTorch                |  COCO     | 2.2.0 |
|  [RepPoints](cv/detection/reppoints_mmdet/pytorch)              |  PyTorch (MMDetection)  |  COCO     | 3.0.0 |
|  [RetinaNet](cv/detection/retinanet/pytorch)                    |  PyTorch                |  COCO     | 2.2.0 |
|  [RetinaNet](cv/detection/retinanet/paddlepaddle)               |  PaddlePaddle           |  COCO     | 3.0.0 |
|  [RT-DETR](cv/detection/rt-detr/pytorch)                        |  PyTorch                |  COCO     | 4.1.1 |
|  [RTMDet](cv/detection/rtmdet/pytorch)                          |  PyTorch                |  COCO     | 3.1.0 |
|  [SSD](cv/detection/ssd/pytorch)                                |  PyTorch                |  COCO     | 2.2.0 |
|  [SSD](cv/detection/ssd/paddlepaddle)                           |  PaddlePaddle           |  COCO     | 2.3.0 |
|  [SSD](cv/detection/ssd/tensorflow)                             |  TensorFlow             |  VOC      | 3.0.0 |
|  [SSD](cv/detection/ssd/mindspore)                              |  MindSpore              |  COCO     | 3.0.0 |
|  [YOLOF](cv/detection/yolof/pytorch)                            |  PyTorch                |  COCO     | 2.2.0 |
|  [YOLOv3](cv/detection/yolov3/pytorch)                          |  PyTorch                |  COCO     | 2.2.0 |
|  [YOLOv3](cv/detection/yolov3/paddlepaddle)                     |  PaddlePaddle           |  COCO     | 2.3.0 |
|  [YOLOv3](cv/detection/yolov3/tensorflow)                       |  TensorFlow             |  VOC      | 3.0.0 |
|  [YOLOv5](cv/detection/yolov5/paddlepaddle)                     |  PaddlePaddle           |  COCO     | 3.1.1 |
|  [YOLOv5](cv/detection/yolov5/pytorch)                          |  PyTorch                |  COCO     | 2.2.0 |
|  [YOLOv6](cv/detection/yolov6/pytorch)                          |  PyTorch                |  COCO     | 3.0.0 |
|  [YOLOv7](cv/detection/yolov7/pytorch)                          |  PyTorch                |  COCO     | 3.0.0 |
|  [YOLOv8](cv/detection/yolov8/pytorch)                          |  PyTorch                |  COCO     | 3.0.0 |
|  [YOLOv9](cv/detection/yolov9/pytorch)                          |  PyTorch                |  COCO     | 4.1.1 |
|  [YOLOv10](cv/detection/yolov10/pytorch)                        |  PyTorch                |  COCO     | 4.1.1 |

#### 三维目标检测

| Model                                                         | Framework | Dataset          | IXUCA SDK |
|---------------------------------------------------------------|-----------|------------------|-------|
|  [BEVFormer](cv/3d_detection/bevformer/pytorch)                 |  PyTorch    |  nuScenes&CAN bus  | 3.0.0 |
|  [CenterPoint](cv/3d_detection/centerpoint/pytorch)             |  PyTorch    |  nuScenes          | 3.1.1 |
|  [PAConv](cv/3d_detection/paconv/pytorch)                       |  PyTorch    |  S3DIS             | 3.1.1 |
|  [Part-A2-Anchor](cv/3d_detection/part_a2_anchor/pytorch)       |  PyTorch    |  KITTI             | 4.1.1 |
|  [Part-A2-Free](cv/3d_detection/part_a2_free/pytorch)           |  PyTorch    |  KITTI             | 4.1.1 |
|  [PointNet++](cv/3d_detection/pointnet2/pytorch)                |  PyTorch    |  S3DIS             | 3.0.0 |
|  [PointPillars](cv/3d_detection/pointpillars/pytorch)           |  PyTorch    |  KITTI             | 3.0.0 |
|  [PointRCNN](cv/3d_detection/pointrcnn/pytorch)                 |  PyTorch    |  KITTI             | 3.1.1 |
|  [PointRCNN-IoU](cv/3d_detection/pointrcnn_iou/pytorch)         |  PyTorch    |  KITTI             | 4.1.1 |
|  [SECOND](cv/3d_detection/second/pytorch)                       |  PyTorch    |  KITTI             | 4.1.1 |
|  [SECOND-IoU](cv/3d_detection/second_iou/pytorch)               |  PyTorch    |  KITTI             | 4.1.1 |

#### 三维重建

| Model                                             | Framework | Dataset | IXUCA SDK |
|---------------------------------------------------|-----------|---------|-------|
|  [HashNeRF](cv/3d-reconstruction/hashnerf/pytorch)  |  PyTorch    |  fox      | 2.2.0 |

#### 图神经网络（GNN）

| Model                                      | Framework    | Dataset                  | IXUCA SDK |
|--------------------------------------------|--------------|--------------------------|-------|
|  [GAT](cv/gnn/gat/paddlepaddle)              |  PaddlePaddle  |  CORA                      | 3.1.0 |
|  [GCN](cv/gnn/GCN/mindspore)                 |  MindSpore     |  CORA & Citeseer           | 3.0.0 |
|  [GCN](cv/gnn/GCN/paddlepaddle)              |  PaddlePaddle  |  CORA & PubMed & Citeseer  | 3.1.0 |
|  [GraphSAGE](cv/gnn/graphsage/paddlepaddle)  |  PaddlePaddle  |  Reddit                    | 3.1.0 |

#### 光学字符识别（OCR）

| Model                                          | Framework    | Dataset   | IXUCA SDK |
|------------------------------------------------|--------------|-----------|-------|
|  [CRNN](cv/ocr/crnn/mindspore)                   |  MindSpore     |  OCR_Recog  | 3.1.0 |
|  [CRNN](cv/ocr/crnn/paddlepaddle)                |  PaddlePaddle  |  LMDB       | 2.3.0 |
|  [DBNet](cv/ocr/dbnet/pytorch)                   |  PyTorch       |  ICDAR2015  | 3.0.0 |
|  [DBNet++](cv/ocr/dbnetpp/paddlepaddle)          |  PaddlePaddle  |  ICDAR2015  | 3.1.1 |
|  [DBNet++](cv/ocr/dbnetpp/pytorch)               |  PyTorch       |  ICDAR2015  | 3.1.0 |
|  [PP-OCR-DB](cv/ocr/pp-ocr-db/paddlepaddle)      |  PaddlePaddle  |  ICDAR2015  | 2.3.0 |
|  [PP-OCR-EAST](cv/ocr/pp-ocr-east/paddlepaddle)  |  PaddlePaddle  |  ICDAR2015  | 3.1.1 |
|  [PSE](cv/ocr/pse/paddlepaddle)                  |  PaddlePaddle  |  OCR_Recog  | 2.3.0 |
|  [SAR](cv/ocr/sar/pytorch)                       |  PyTorch       |  OCR_Recog  | 2.2.0 |
|  [SAST](cv/ocr/sast/paddlepaddle)                |  PaddlePaddle  |  ICDAR2015  | 3.1.1 |
|  [SATRN](cv/ocr/satrn/pytorch/base)              |  PyTorch       |  OCR_Recog  | 2.2.0 |

#### 点云

| Model                                           | Framework | Dataset                         | IXUCA SDK |
|-------------------------------------------------|-----------|---------------------------------|-------|
|  [Point-BERT](cv/point_cloud/point-bert/pytorch)  |  PyTorch    |  ShapeNet55 & processed ModelNet  | 2.2.0 |

#### 姿态估计

| Model                                   | Framework    | Dataset | IXUCA SDK |
|-----------------------------------------|--------------|---------|-------|
|  [AlphaPose](cv/pose/alphapose/pytorch)   |  PyTorch       |  COCO     | 3.0.0 |
|  [HRNet](cv/pose/hrnet/pytorch)           |  PyTorch       |  COCO     | 2.2.0 |
|  [HRNet-W32](cv/pose/hrnet/paddlepaddle)  |  PaddlePaddle  |  COCO     | 3.1.0 |
|  [OpenPose](cv/pose/openpose/mindspore)   |  MindSpore     |  COCO     | 3.1.0 |

#### 自监督学习

| Model                                          | Framework | Dataset  | IXUCA SDK |
|------------------------------------------------|-----------|----------|-------|
|  [MAE](cv/self_supervised_learning/mae/pytorch)  |  PyTorch    |  ImageNet  | 3.0.0 |

#### 语义分割

| Model                                                                | Framework    | Dataset        | IXUCA SDK |
|----------------------------------------------------------------------|--------------|----------------|-------|
|  [3D-UNet](cv/semantic_segmentation/unet3d/pytorch)                    |  PyTorch       |  kits19          | 2.2.0 |
|  [APCNet](cv/semantic_segmentation/apcnet/pytorch)                     |  PyTorch       |  Cityscapes      | 2.2.0 |
|  [Attention U-net](cv/semantic_segmentation/att_unet/pytorch)          |  PyTorch       |  Cityscapes      | 3.0.0 |
|  [BiSeNet](cv/semantic_segmentation/bisenet/pytorch)                   |  PyTorch       |  COCO            | 2.2.0 |
|  [BiSeNetV2](cv/semantic_segmentation/bisenetv2/paddlepaddle)          |  PaddlePaddle  |  Cityscapes      | 3.0.0 |
|  [BiSeNetV2](cv/semantic_segmentation/bisenetv2/pytorch)               |  PyTorch       |  Cityscapes      | 3.1.1 |
|  [CGNet](cv/semantic_segmentation/cgnet/pytorch)                       |  PyTorch       |  COCO            | 2.2.0 |
|  [ContextNet](cv/semantic_segmentation/contextnet/pytorch)             |  PyTorch       |  COCO            | 2.2.0 |
|  [DabNet](cv/semantic_segmentation/dabnet/pytorch)                     |  PyTorch       |  COCO            | 2.2.0 |
|  [DANet](cv/semantic_segmentation/danet/pytorch)                       |  PyTorch       |  COCO            | 2.2.0 |
|  [DDRnet](cv/semantic_segmentation/ddrnet/pytorch)                     |  PyTorch       |  Cityscapes      | 3.0.0 |
|  [DeepLabV3](cv/semantic_segmentation/deeplabv3/pytorch)               |  PyTorch       |  COCO            | 2.2.0 |
|  [DeepLabV3](cv/semantic_segmentation/deeplabv3/paddlepaddle)          |  PaddlePaddle  |  Cityscapes      | 2.3.0 |
|  [DeepLabV3](cv/semantic_segmentation/deeplabv3/mindspore)             |  MindSpore     |  VOC             | 3.0.0 |
|  [DeepLabV3+](cv/semantic_segmentation/deeplabv3plus/paddlepaddle)     |  PaddlePaddle  |  Cityscapes      | 3.0.0 |
|  [DeepLabV3+](cv/semantic_segmentation/deeplabv3plus/tensorflow)       |  TensorFlow    |  Cityscapes      | 3.1.0 |
|  [DenseASPP](cv/semantic_segmentation/denseaspp/pytorch)               |  PyTorch       |  COCO            | 2.2.0 |
|  [DFANet](cv/semantic_segmentation/dfanet/pytorch)                     |  PyTorch       |  COCO            | 2.2.0 |
|  [DNLNet](cv/semantic_segmentation/dnlnet/paddlepaddle)                |  PaddlePaddle  |  Cityscapes      | 2.3.0 |
|  [DUNet](cv/semantic_segmentation/dunet/pytorch)                       |  PyTorch       |  COCO            | 2.2.0 |
|  [EncNet](cv/semantic_segmentation/encnet/pytorch)                     |  PyTorch       |  COCO            | 2.2.0 |
|  [ENet](cv/semantic_segmentation/enet/pytorch)                         |  PyTorch       |  COCO            | 2.2.0 |
|  [ERFNet](cv/semantic_segmentation/erfnet/pytorch)                     |  PyTorch       |  COCO            | 2.2.0 |
|  [ESPNet](cv/semantic_segmentation/espnet/pytorch)                     |  PyTorch       |  COCO            | 2.2.0 |
|  [FastFCN](cv/semantic_segmentation/fastfcn/paddlepaddle)              |  PyTorch       |  ADE20K          | 3.1.0 |
|  [FastSCNN](cv/semantic_segmentation/fastscnn/pytorch)                 |  PyTorch       |  COCO            | 2.2.0 |
|  [FCN](cv/semantic_segmentation/fcn/pytorch)                           |  PyTorch       |  COCO            | 2.2.0 |
|  [FPENet](cv/semantic_segmentation/fpenet/pytorch)                     |  PyTorch       |  COCO            | 2.2.0 |
|  [GCNet](cv/semantic_segmentation/gcnet/pytorch)                       |  PyTorch       |  Cityscapes      | 2.2.0 |
|  [HardNet](cv/semantic_segmentation/hardnet/pytorch)                   |  PyTorch       |  COCO            | 2.2.0 |
|  [ICNet](cv/semantic_segmentation/icnet/pytorch)                       |  PyTorch       |  COCO            | 2.2.0 |
|  [LedNet](cv/semantic_segmentation/lednet/pytorch)                     |  PyTorch       |  COCO            | 2.2.0 |
|  [LinkNet](cv/semantic_segmentation/linknet/pytorch)                   |  PyTorch       |  COCO            | 2.2.0 |
|  [Mask2Former](cv/semantic_segmentation/mask2former/pytorch)           |  PyTorch       |  Cityscapes      | 3.1.0 |
|  [MobileSeg](cv/semantic_segmentation/mobileseg/paddlepaddle)          |  PaddlePaddle  |  Cityscapes      | 3.1.0 |
|  [OCNet](cv/semantic_segmentation/ocnet/pytorch)                       |  PyTorch       |  COCO            | 2.2.0 |
|  [OCRNet](cv/semantic_segmentation/ocrnet/paddlepaddle)                |  PaddlePaddle  |  Cityscapes      | 3.1.0 |
|  [OCRNet](cv/semantic_segmentation/ocrnet/pytorch)                     |  PyTorch       |  Cityscapes      | 2.2.0 |
|  [PP-HumanSegV1](cv/semantic_segmentation/pp_humansegv1/paddlepaddle)  |  PaddlePaddle  |  PP-HumanSeg14K  | 3.1.0 |
|  [PP-HumanSegV2](cv/semantic_segmentation/pp_humansegv2/paddlepaddle)  |  PaddlePaddle  |  PP-HumanSeg14K  | 3.1.0 |
|  [PP-LiteSeg](cv/semantic_segmentation/pp_liteseg/paddlepaddle)        |  PaddlePaddle  |  Cityscapes      | 3.1.0 |
|  [PSANet](cv/semantic_segmentation/psanet/pytorch)                     |  PyTorch       |  COCO            | 2.2.0 |
|  [PSPNet](cv/semantic_segmentation/pspnet/pytorch)                     |  PyTorch       |  Cityscapes      | 2.2.0 |
|  [RefineNet](cv/semantic_segmentation/refinenet/pytorch)               |  PyTorch       |  COCO            | 2.2.0 |
|  [SegNet](cv/semantic_segmentation/segnet/pytorch)                     |  PyTorch       |  COCO            | 2.2.0 |
|  [STDC](cv/semantic_segmentation/stdc/paddlepaddle)                    |  PaddlePaddle  |  Cityscapes      | 3.1.0 |
|  [STDC](cv/semantic_segmentation/stdc/pytorch)                         |  PyTorch       |  Cityscapes      | 3.0.0 |
|  [UNet](cv/semantic_segmentation/unet/pytorch)                         |  PyTorch       |  COCO            | 2.2.0 |
|  [UNet](cv/semantic_segmentation/unet/paddlepaddle)                    |  PaddlePaddle  |  Cityscapes      | 2.3.0 |
|  [UNet++](cv/semantic_segmentation/unet++/pytorch)                     |  PyTorch       |  DRIVE           | 3.0.0 |
|  [VNet](cv/semantic_segmentation/vnet/tensorflow)                      |  TensorFlow    |  Hippocampus     | 3.0.0 |

#### 超分辨率

| Model                                                     | Framework | Dataset | IXUCA SDK |
|-----------------------------------------------------------|-----------|---------|-------|
|  [basicVSR++](cv/super_resolution/basicvsr++/pytorch)       |  PyTorch    |  REDS     | 2.2.0 |
|  [basicVSR](cv/super_resolution/basicvsr/pytorch)           |  PyTorch    |  REDS     | 2.2.0 |
|  [ESRGAN](cv/super_resolution/esrgan/pytorch)               |  PyTorch    |  DIV2K    | 2.2.0 |
|  [LIIF](cv/super_resolution/liif/pytorch)                   |  PyTorch    |  DIV2K    | 2.2.0 |
|  [RealBasicVSR](cv/super_resolution/real_basicvsr/pytorch)  |  PyTorch    |  REDS     | 2.2.0 |
|  [TTSR](cv/super_resolution/ttsr/pytorch)                   |  PyTorch    |  CUFED    | 2.2.0 |
|  [TTVSR](cv/super_resolution/ttvsr/pytorch)                 |  PyTorch    |  REDS     | 2.2.0 |

#### 多目标跟踪

| Model                                           | Framework    | Dataset     | IXUCA SDK |
|-------------------------------------------------|--------------|-------------|-------|
|  [ByteTrack](cv/multi_object_tracking/bytetrack/paddlepaddle)  |  PaddlePaddle  |  MOT17        | 3.1.0 |
|  [DeepSORT](cv/multi_object_tracking/deep_sort/pytorch)        |  PyTorch       |  Market-1501  | 3.0.0 |
|  [FairMOT](cv/multi_object_tracking/fairmot/pytorch)           |  PyTorch       |  MOT17        | 2.2.0 |

### 多模态

| Model                                                                                       | Framework | Dataset        | IXUCA SDK |
|---------------------------------------------------------------------------------------------|-----------|----------------|-----------|
| [BLIP](multimodal/vision-language_model/blip/pytorch)                                       | PyTorch   | COCO           | 3.1.1     |
| [CLIP](multimodal/contrastive_learning/clip/pytorch)                                        | PyTorch   | CIFAR100       | 2.2.0     |
| [ControlNet](multimodal/diffusion_model/controlnet/pytorch)                                 | PyTorch   | Fill50K        | 3.1.0     |
| [DDPM](multimodal/diffusion_model/ddpm/pytorch)                                             | PyTorch   | CIFAR-10       | 3.1.0     |
| [LLaVA 1.5](multimodal/vision-language_model/llava-1.5/pytorch)                             | PyTorch   | LLaVA-Pretrain | 4.1.1     |
| [L-Verse](multimodal/vision-language_model/l-verse/pytorch)                                 | PyTorch   | ImageNet       | 2.2.0     |
| [MoE-LLaVA-Phi2-2.7B](multimodal/vision-language_model/moe-llava-phi2-2.7b/pytorch)         | PyTorch   | MoE-LLaVA      | 4.2.0     |
| [MoE-LLaVA-Qwen-1.8B](multimodal/vision-language_model/moe-llava-qwen-1.8b/pytorch)         | PyTorch   | MoE-LLaVA      | 4.2.0     |
| [MoE-LLaVA-StableLM-1.6B](multimodal/vision-language_model/moe-llava-stablelm-1.6b/pytorch) | PyTorch   | MoE-LLaVA      | 4.2.0     |
| [Stable Diffusion 1.4](multimodal/diffusion_model/stable-diffusion-1.4/pytorch)             | PyTorch   | pokemon-images | 3.0.0     |
| [Stable Diffusion 1.5](multimodal/diffusion_model/stable-diffusion-1.5/pytorch)             | PyTorch   | pokemon-images | 4.1.1     |
| [Stable Diffusion 2.1](multimodal/diffusion_model/stable-diffusion-2.1/pytorch)             | PyTorch   | pokemon-images | 4.1.1     |
| [Stable Diffusion 3](multimodal/diffusion_model/stable-diffusion-3/pytorch)                 | PyTorch   | dog-example    | 4.1.1     |
| [Stable Diffusion XL](multimodal/diffusion_model/stable-diffusion-xl/pytorch)               | PyTorch   | pokemon-images | 4.1.1     |

### 自然语言处理（NLP）

#### 完形填空

| Model                                                   | Framework | Dataset               | IXUCA SDK |
|---------------------------------------------------------|-----------|-----------------------|-------|
|  [GLM](nlp/cloze_test/glm/pytorch)  |  PyTorch    |  GLMForMultiTokenCloze  | 2.2.0 |

#### 对话生成

| Model                                      | Framework | Dataset | IXUCA SDK |
|--------------------------------------------|-----------|---------|-------|
|  [CPM](nlp/dialogue_generation/cpm/pytorch)  |  PyTorch    |  STC      | 2.2.0 |

#### 语言建模

| Model                                                            | Framework         | Dataset            | IXUCA SDK |
|------------------------------------------------------------------|-------------------|--------------------|-------|
|  [BART](nlp/language_model/bart_fairseq/pytorch)                   |  PyTorch (Fairseq)  |  RTE                 | 3.0.0 |
|  [BERT NER](nlp/ner/bert/pytorch)                                  |  PyTorch            |  CoNLL-2003          | 3.0.0 |
|  [BERT Pretraining](nlp/language_model/bert/pytorch)               |  PyTorch            |  MLCommon Wikipedia  | 2.2.0 |
|  [BERT Pretraining](nlp/language_model/bert/paddlepaddle)          |  PaddlePaddle       |  MNLI                | 2.3.0 |
|  [BERT Pretraining](nlp/language_model/bert/tensorflow)            |  TensorFlow         |  MNLI                | 3.0.0 |
|  [BERT Pretraining](nlp/language_model/bert/mindspore)             |  MindSpore          |  SQuAD               | 3.0.0 |
|  [BERT Text Classification](nlp/text_classification/bert/pytorch)  |  PyTorch            |  GLUE                | 3.0.0 |
|  [BERT Text Summerization](nlp/text_summarisation/bert/pytorch)    |  PyTorch            |  cnn_dailymail       | 3.0.0 |
|  [BERT Question Answering](nlp/question_answering/bert/pytorch)    |  PyTorch            |  SQuAD               | 3.0.0 |
|  [GPT2-Medium-EN](nlp/llm/gpt2-medium-en/paddlepaddle)             |  PaddlePaddle       |  SST-2               | 3.1.0 |
|  [RoBERTa](nlp/language_model/roberta_fairseq/pytorch)             |  PyTorch (Fairseq)  |  RTE                 | 3.0.0 |
|  [XLNet](nlp/language_model/xlnet/paddlepaddle)                    |  PaddlePaddle       |  SST-2               | 3.1.0 |

#### 文本纠错

| Model                                           | Framework    | Dataset | IXUCA SDK |
|-------------------------------------------------|--------------|---------|-------|
|  [ERNIE](nlp/text_correction/ernie/paddlepaddle)  |  PaddlePaddle  |  corpus   | 2.3.0 |

#### 翻译

| Model                                                          | Framework         | Dataset | IXUCA SDK |
|----------------------------------------------------------------|-------------------|---------|-------|
|  [Convolutional](nlp/translation/convolutional_fairseq/pytorch)  |  PyTorch (Fairseq)  |  WMT14    | 3.0.0 |
|  [T5](nlp/translation/t5/pytorch)                                |  PyTorch            |  WMT14    | 2.2.0 |
|  [Transformer](nlp/translation/transformer/paddlepaddle)         |  PaddlePaddle       |  WMT14    | 2.3.0 |
|  [Transformer](nlp/translation/transformer_fairseq/pytorch)      |  PyTorch (Fairseq)  |  IWSLT14  | 3.0.0 |

### 强化学习

| Model                                                              | Framework    | Dataset     | IXUCA SDK |
|--------------------------------------------------------------------|--------------|-------------|-------|
|  [DQN](reinforcement_learning/q-learning-networks/dqn/paddlepaddle)  |  PaddlePaddle  |  CartPole-v0  | 3.1.0 |

### 语音

#### 语音识别

| Model                                                                                   | Framework       | Dataset  | IXUCA SDK |
|-----------------------------------------------------------------------------------------|-----------------|----------|-------|
|  [Conformer](audio/speech_recognition/conformer_wenet/pytorch)                            |  PyTorch (WeNet)  |  AISHELL   | 2.2.0 |
|  [Efficient Conformer v2](audio/speech_recognition/efficient_conformer_v2_wenet/pytorch)  |  PyTorch (WeNet)  |  AISHELL   | 3.1.0 |
|  [PP-ASR-Conformer](audio/speech_recognition/conformer/paddlepaddle)                      |  PaddlePaddle     |  AISHELL   | 3.1.0 |
|  [RNN-T](audio/speech_recognition/rnnt/pytorch)                                           |  PyTorch          |  LJSpeech  | 2.2.0 |
|  [Transformer](audio/speech_recognition/transformer_wenet/pytorch)                        |  PyTorch (WeNet)  |  AISHELL   | 3.0.0 |
|  [U2++ Conformer](audio/speech_recognition/u2++_conformer_wenet/pytorch)                  |  PyTorch (WeNet)  |  AISHELL   | 3.0.0 |
|  [Unified Conformer](audio/speech_recognition/unified_conformer_wenet/pytorch)            |  PyTorch (WeNet)  |  AISHELL   | 3.0.0 |

#### 语音合成

| Model                                                                 | Framework    | Dataset     | IXUCA SDK |
|-----------------------------------------------------------------------|--------------|-------------|-------|
|  [PP-TTS-FastSpeech2](audio/speech_synthesis/fastspeech2/paddlepaddle)  |  PaddlePaddle  |  CSMSC        | 3.1.0 |
|  [PP-TTS-HiFiGAN](audio/speech_synthesis/hifigan/paddlepaddle)          |  PaddlePaddle  |  CSMSC        | 3.1.0 |
|  [Tacotron2](audio/speech_synthesis/tacotron2/pytorch)                  |  PyTorch       |  LJSpeech     | 2.2.0 |
|  [VQMIVC](audio/speech_synthesis/vqmivc/pytorch)                        |  PyTorch       |  VCTK-Corpus  | 2.2.0 |
|  [WaveGlow](audio/speech_synthesis/waveglow/pytorch)                    |  PyTorch       |  LJSpeech     | 2.2.0 |

### 其他

#### 图机器学习

| Model                                                                | Framework | Dataset            | IXUCA SDK |
|----------------------------------------------------------------------|-----------|--------------------|-------|
|  [Graph WaveNet](others/graph_machine_learning/graph_wavenet/pytorch)  |  PyTorch    |  METR-LA & PEMS-BAY  | 2.2.0 |

#### 柯尔莫哥洛夫-阿诺德网络（KAN）

| Model                                                | Framework | Dataset | IXUCA SDK |
|------------------------------------------------------|-----------|---------|-------|
|  [KAN](others/kolmogorov_arnold_networks/kan/pytorch)  |  PyTorch    |  -        | 4.1.1 |

#### 模型剪枝

| Model                                                             | Framework | Dataset      | IXUCA SDK |
|-------------------------------------------------------------------|-----------|--------------|-------|
|  [Network Slimming](others/model_pruning/network-slimming/pytorch)  |  PyTorch    |  CIFAR-10/100  | 3.0.0 |

#### 推荐系统

| Model                                                             | Framework    | Dataset         | IXUCA SDK |
|-------------------------------------------------------------------|--------------|-----------------|-------|
|  [DeepFM](others/recommendation_systems/deepfm/paddlepaddle)        |  PaddlePaddle  |  Criteo_Terabyte  | 2.3.0 |
|  [DLRM](others/recommendation_systems/dlrm/pytorch)                 |  PyTorch       |  Criteo_Terabyte  | 2.2.0 |
|  [DLRM](others/recommendation_systems/dlrm/paddlepaddle)            |  PaddlePaddle  |  Criteo_Terabyte  | 3.1.0 |
|  [FFM](others/recommendation_systems/ffm/paddlepaddle)              |  PaddlePaddle  |  Criteo_Terabyte  | 3.1.0 |
|  [NCF](others/recommendation_systems/ncf/pytorch)                   |  PyTorch       |  movielens        | 2.2.0 |
|  [Wide&Deep](others/recommendation_systems/wide_deep/paddlepaddle)  |  PaddlePaddle  |  Criteo_Terabyte  | 2.3.0 |
|  [xDeepFM](others/recommendation_systems/xdeepfm/paddlepaddle)      |  PaddlePaddle  |  Criteo_Terabyte  | 3.1.0 |

--------

## 容器镜像构建方式

社区用户可参考[容器镜像构建说明](docker/Iluvatar/README.md)在本地构建出能够运行DeepSparkHub仓库中模型的容器镜像。

--------

## 社区

### 治理

请参见 DeepSpark Code of Conduct on [Gitee](https://gitee.com/deep-spark/deepspark/blob/master/CODE_OF_CONDUCT.md) or on
[GitHub](https://github.com/Deep-Spark/deepspark/blob/main/CODE_OF_CONDUCT.md)。

### 交流

请联系 <contact@deepspark.org.cn>。

### 贡献

请参见 [DeepSparkHub Contributing Guidelines](CONTRIBUTING.md)。

### 免责声明

DeepSparkHub仅提供公共数据集的下载和预处理脚本。这些数据集不属于DeepSparkHub，DeepSparkHub也不对其质量或维护负责。请确保
您具有这些数据集的使用许可，基于这些数据集训练的模型仅可用于非商业研究和教育。

致数据集所有者：

如果不希望您的数据集公布在DeepSparkHub上或希望更新DeepSparkHub中属于您的数据集，请在Gitee或Github上提交issue，我们将按您
的issue删除或更新。衷心感谢您对我们社区的支持和贡献。

## 许可证

本项目许可证遵循[Apache-2.0](LICENSE)。
