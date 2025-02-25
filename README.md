# DeepSparkHub

DeepSparkHub甄选上百个应用算法和模型，覆盖AI和通用计算各领域，支持主流市场智能计算场景，包括智慧城市、数字个人、医疗、教
育、通信、能源等多个领域。

## 模型列表

### LLM (Large Language Model)

| Model                                                       | Framework | ToolBox            | Dataset/Weight        |
|-------------------------------------------------------------|-----------|--------------------|-----------------------|
| [Aquila2-34B](nlp/llm/aquila2-34b/megatron-deepspeed)       | PyTorch   | Megatron-DeepSpeed | Bookcorpus            |
| [Baichuan2-7B](nlp/llm/baichuan2-7b/Baichuan2)              | PyTorch   | DeepSpeed          | baichuan2-7b-base     |
| [Bloom-7B1](nlp/llm/bloom-7b1/firefly)                      | PyTorch   | Firefly            | school_math_0.25M     |
| [ChatGLM-6B](nlp/llm/chatglm-6b/deepspeed)                  | PyTorch   | DeepSpeed          | ADGEN & chatglm-6b    |
| [ChatGLM2-6B SFT](nlp/llm/ChatGLM2-6b-sft)                  | PyTorch   | DeepSpeed          | ADGEN & chatglm2-6b   |
| [ChatGLM3-6B](nlp/llm/chatglm3-6b/deepspeed/finetune_demo)  | PyTorch   | DeepSpeed          | ADGEN & chatglm3-6b   |
| [DeepSeekMoE 7B](nlp/llm/deepseek_moe_7b/colossalai)        | PyTorch   | ColossalAI         | deepseek-moe-16b-base |
| [Llama-7B](nlp/llm/llama-7b/colossalai)                     | PyTorch   | ColossalAI         | llama-7b-hf           |
| [Llama2-7B](nlp/llm/llama2-7b/megatron-deepspeed)           | PyTorch   | Megatron-DeepSpeed | Bookcorpus            |
| [Llama2-7B RMF](nlp/llm/llama2-7b_reward_sft/deepspeed)     | PyTorch   | DeepSpeed          | Dahoas/rm-static      |
| [Llama2-7B RLHF](nlp/llm/llama2-7b_rlhf/megatron-deepspeed) | PyTorch   | Megatron-DeepSpeed | llama2-7b&tiny-llama  |
| [Llama2-7B SFT](nlp/llm/llama2-7b_sft/megatron-deepspeed)   | PyTorch   | Megatron-DeepSpeed | GPT Small-117M        |
| [Llama2-13B](nlp/llm/llama2-13b/megatron-deepspeed)         | PyTorch   | Megatron-DeepSpeed | Bookcorpus            |
| [Llama2-34B](nlp/llm/llama2-34b/megatron-deepspeed)         | PyTorch   | Megatron-DeepSpeed | Bookcorpus            |
| [Llama3-8B](nlp/llm/llama3_8b/megatron-deepspeed)           | PyTorch   | Megatron-DeepSpeed | Bookcorpus            |
| [Llama3-8B SFT](nlp/llm/llama3_8b/colossalai)               | PyTorch   | ColossalAI         | school_math_0.25M     |
| [Mamba-2](nlp/llm/mamba-2/megatron-lm)                      | PyTorch   | Megatron-LM        | GPT Small-117M        |
| [Mixtral 8x7B](nlp/llm/mixtral/megatron-lm)                 | PyTorch   | Megatron-LM        | GPT Small-117M        |
| [QWen-7B](nlp/llm/qwen-7b/firefly)                          | PyTorch   | Firefly            | qwen-7b               |
| [QWen1.5-7B](nlp/llm/qwen1.5-7b/firefly)                    | PyTorch   | Firefly            | school_math           |
| [QWen1.5-14B](nlp/llm/qwen1.5-14b/firefly)                  | PyTorch   | Firefly            | school_math           |
| [Qwen2.5-7B SFT](nlp/llm/qwen2.5-7b/LLaMA-Factory)          | PyTorch   | LLaMA-Factory      | qwen2.5-7b            |

### Computer Vision

#### Classification

| Model                                                                         | Framework    | Dataset  |
|-------------------------------------------------------------------------------|--------------|----------|
| [ACmix](cv/classification/acmix/pytorch)                                      | PyTorch      | ImageNet |
| [ACNet](cv/classification/acnet/pytorch)                                      | PyTorch      | ImageNet |
| [AlexNet](cv/classification/alexnet/pytorch)                                  | PyTorch      | ImageNet |
| [AlexNet](cv/classification/alexnet/tensorflow)                               | TensorFlow   | ImageNet |
| [BYOL](cv/classification/byol/pytorch)                                        | PyTorch      | ImageNet |
| [CBAM](cv/classification/cbam/pytorch)                                        | PyTorch      | ImageNet |
| [ConvNext](cv/classification/convnext/pytorch)                                | PyTorch      | ImageNet |
| [CspDarknet53](cv/classification/cspdarknet53/pytorch)                        | PyTorch      | ImageNet |
| [DenseNet](cv/classification/densenet/paddlepaddle)                           | PaddlePaddle | ImageNet |
| [DenseNet](cv/classification/densenet/pytorch)                                | PyTorch      | ImageNet |
| [DPN92](cv/classification/dpn92/pytorch)                                      | PyTorch      | ImageNet |
| [DPN107](cv/classification/dpn107/pytorch)                                    | PyTorch      | ImageNet |
| [ECA-MobileNetV2](cv/classification/eca_mobilenet_v2/pytorch)                 | PyTorch      | ImageNet |
| [ECA-ResNet152](cv/classification/eca_resnet152/pytorch)                      | PyTorch      | ImageNet |
| [EfficientNetB0](cv/classification/efficientnet_b0/paddlepaddle)              | PaddlePaddle | ImageNet |
| [EfficientNetB4](cv/classification/efficientnet_b4/pytorch)                   | PyTorch      | ImageNet |
| [FasterNet](cv/classification/fasternet/pytorch)                              | PyTorch      | ImageNet |
| [GoogLeNet](cv/classification/googlenet/pytorch)                              | PyTorch      | ImageNet |
| [GoogLeNet](cv/classification/googlenet/paddlepaddle)                         | PaddlePaddle | ImageNet |
| [InceptionV3](cv/classification/inceptionv3/mindspore)                        | MindSpore    | ImageNet |
| [InceptionV3](cv/classification/inceptionv3/pytorch)                          | PyTorch      | ImageNet |
| [InceptionV3](cv/classification/inceptionv3/tensorflow)                       | TensorFlow   | ImageNet |
| [InceptionV4](cv/classification/inceptionv4/pytorch)                          | PyTorch      | ImageNet |
| [InternImage](cv/classification/internimage/pytorch)                          | PyTorch      | ImageNet |
| [LeNet](cv/classification/lenet/pytorch)                                      | PyTorch      | ImageNet |
| [MobileNetV2](cv/classification/mobilenetv2/pytorch)                          | PyTorch      | ImageNet |
| [MobileNetV3](cv/classification/mobilenetv3/mindspore)                        | MindSpore    | ImageNet |
| [MobileNetV3](cv/classification/mobilenetv3/pytorch)                          | PyTorch      | ImageNet |
| [MobileNetV3](cv/classification/mobilenetv3/paddlepaddle)                     | PaddlePaddle | ImageNet |
| [MobileNetV3_Large1.0](cv/classification/mobilenetv3_large_x1_0/paddlepaddle) | PaddlePaddle | ImageNet |
| [MobileOne](cv/classification/mobileone/pytorch)                              | PyTorch      | ImageNet |
| [MoCoV2](cv/classification/mocov2/pytorch)                                    | PyTorch      | ImageNet |
| [PP-LCNet](cv/classification/pp-lcnet/paddlepaddle)                           | PaddlePaddle | ImageNet |
| [RepMLP](cv/classification/repmlp/pytorch)                                    | PyTorch      | ImageNet |
| [RepVGG](cv/classification/repvgg/pytorch)                                    | PyTorch      | ImageNet |
| [RepVGG](cv/classification/repvgg/paddlepaddle)                               | PaddlePaddle | ImageNet |
| [RepViT](cv/classification/repvit/pytorch)                                    | PyTorch      | ImageNet |
| [Res2Net50_14w_8s](cv/classification/Res2Net50_14w_8s/paddlepaddle)           | PaddlePaddle | ImageNet |
| [ResNeSt14](cv/classification/resnest14/pytorch)                              | PyTorch      | ImageNet |
| [ResNeSt50](cv/classification/resnest50/pytorch)                              | PyTorch      | ImageNet |
| [ResNeSt50](cv/classification/resnest50/paddlepaddle)                         | PaddlePaddle | ImageNet |
| [ResNeSt101](cv/classification/resnest101/pytorch)                            | PyTorch      | ImageNet |
| [ResNeSt269](cv/classification/resnest269/pytorch)                            | PyTorch      | ImageNet |
| [ResNet18](cv/classification/resnet18/pytorch)                                | PyTorch      | ImageNet |
| [ResNet50](cv/classification/resnet50/pytorch)                                | PyTorch      | ImageNet |
| [ResNet50](cv/classification/resnet50/paddlepaddle)                           | PaddlePaddle | ImageNet |
| [ResNet50](cv/classification/resnet50/tensorflow)                             | TensorFlow   | ImageNet |
| [ResNet101](cv/classification/resnet101/pytorch)                              | PyTorch      | ImageNet |
| [ResNet152](cv/classification/resnet152/pytorch)                              | PyTorch      | ImageNet |
| [ResNeXt50_32x4d](cv/classification/resnext50_32x4d/mindspore)                | MindSpore    | ImageNet |
| [ResNeXt50_32x4d](cv/classification/resnext50_32x4d/pytorch)                  | PyTorch      | ImageNet |
| [ResNeXt101_32x8d](cv/classification/resnext101_32x8d/pytorch)                | PyTorch      | ImageNet |
| [SE_ResNet50_vd](cv/classification/se_resnet50_vd/paddlepaddle)               | PaddlePaddle | ImageNet |
| [SEResNeXt](cv/classification/seresnext/pytorch)                              | PyTorch      | ImageNet |
| [ShuffleNetV2](cv/classification/shufflenetv2/paddlepaddle)                   | PaddlePaddle | ImageNet |
| [ShuffleNetV2](cv/classification/shufflenetv2/pytorch)                        | PyTorch      | ImageNet |
| [SqueezeNet](cv/classification/squeezenet/pytorch)                            | PyTorch      | ImageNet |
| [Swin Transformer](cv/classification/swin_transformer/paddlepaddle)           | PaddlePaddle | ImageNet |
| [Swin Transformer](cv/classification/swin_transformer/pytorch)                | PyTorch      | ImageNet |
| [VGG16](cv/classification/vgg/paddlepaddle)                                   | PaddlePaddle | ImageNet |
| [VGG16](cv/classification/vgg/pytorch)                                        | PyTorch      | ImageNet |
| [VGG16](cv/classification/vgg/tensorflow)                                     | TensorFlow   | ImageNet |
| [Wave-MLP](cv/classification/wavemlp/pytorch)                                 | PyTorch      | ImageNet |
| [Wide_ResNet101_2](cv/classification/wide_resnet101_2/pytorch)                | PyTorch      | ImageNet |
| [Xception](cv/classification/xception/paddlepaddle)                           | PaddlePaddle | ImageNet |
| [Xception](cv/classification/xception/pytorch)                                | PyTorch      | ImageNet |

#### Face Detection

| Model                                              | Framework | Dataset    |
|----------------------------------------------------|-----------|------------|
| [RetinaFace](cv/face_detection/retinaface/pytorch) | PyTorch   | WIDER FACE |

#### Face Recognition

| Model                                                   | Framework    | Dataset            |
|---------------------------------------------------------|--------------|--------------------|
| [ArcFace](cv/face_recognition//arcface/pytorch)         | PyTorch      | CASIA-WebFaces&LFW |
| [BlazeFace](cv/face_recognition/blazeface/paddlepaddle) | PaddlePaddle | WIDER FACE         |
| [CosFace](cv/face_recognition/cosface/pytorch)          | PyTorch      | CASIA-WebFaces&LFW |
| [FaceNet](cv/face_recognition/facenet/pytorch)          | PyTorch      | CASIA-WebFaces&LFW |
| [FaceNet](cv/face_recognition/facenet/tensorflow)       | TensorFlow   | CASIA-WebFaces&LFW |

#### Instance Segmentation

| Model                                               | Framework    | Dataset |
|-----------------------------------------------------|--------------|---------|
| [SOLO](cv/instance_segmentation/SOLO/pytorch)       | PyTorch      | COCO    |
| [SOLOv2](cv/detection/solov2/paddlepaddle)          | PaddlePaddle | COCO    |
| [SOLOv2](cv/instance_segmentation/solov2/pytorch)   | PyTorch      | COCO    |
| [YOLACT++](cv/instance_segmentation/yolact/pytorch) | PyTorch      | COCO    |

#### Image Generation

| Model                                               | Framework    | Dataset  |
|-----------------------------------------------------|--------------|----------|
| [DCGAN](cv/image_generation/dcgan/mindspore)        | MindSpore    | ImageNet |
| [Pix2Pix](cv/image_generation/Pix2pix/paddlepaddle) | PaddlePaddle | facades  |

#### Knowledge Distillation

| Model                             | Framework | Dataset      |
|-----------------------------------|-----------|--------------|
| [CWD](cv/distiller/CWD/pytorch)   | PyTorch   | Cityscapes   |
| [RKD](cv/distiller/RKD/pytorch)   | PyTorch   | CUB-200-2011 |
| [WSLD](cv/distiller/WSLD/pytorch) | PyTorch   | ImageNet     |

#### Object Detection

| Model                                                         | Framework             | Dataset |
|---------------------------------------------------------------|-----------------------|---------|
| [ATSS](cv/detection/atss_mmdet/pytorch)                       | PyTorch (MMDetection) | COCO    |
| [AutoAssign](cv/detection/autoassign/pytorch)                 | PyTorch               | COCO    |
| [Cascade R-CNN](cv/detection/cascade_rcnn_mmdet/pytorch)      | PyTorch (MMDetection) | COCO    |
| [CenterMask2](cv/detection/centermask2/pytorch)               | PyTorch               | COCO    |
| [CenterNet](cv/detection/centernet/pytorch)                   | PyTorch               | COCO    |
| [CenterNet](cv/detection/centernet/paddlepaddle)              | PaddlePaddle          | COCO    |
| [Co-DETR](cv/detection/co-detr/pytorch)                       | PyTorch               | COCO    |
| [CornerNet](cv/detection/cornernet_mmdet/pytorch)             | PyTorch (MMDetection) | COCO    |
| [DCNV2](cv/detection/dcnv2_mmdet/pytorch)                     | PyTorch (MMDetection) | COCO    |
| [DETR](cv/detection/detr/paddlepaddle)                        | PaddlePaddle          | COCO    |
| [Faster R-CNN](cv/detection/fasterrcnn/pytorch)               | PyTorch               | COCO    |
| [FCOS](cv/detection/fcos/paddlepaddle)                        | PaddlePaddle          | COCO    |
| [FCOS](cv/detection/fcos/pytorch)                             | PyTorch               | COCO    |
| [Mamba-YOLO](cv/detection/mamba_yolo/pytorch)                 | PyTorch               | COCO    |
| [Mask R-CNN](cv/detection/maskrcnn/pytorch)                   | PyTorch               | COCO    |
| [Mask R-CNN](cv/detection/maskrcnn/paddlepaddle)              | PaddlePaddle          | COCO    |
| [OC_SORT](cv/detection/oc_sort/paddlepaddle)                  | PaddlePaddle          | MOT17   |
| [Oriented RepPoints](cv/detection/oriented_reppoints/pytorch) | PyTorch               | DOTA    |
| [PP-PicoDet](cv/detection/picodet/paddlepaddle)               | PaddlePaddle          | COCO    |
| [PP-YOLOE](cv/detection/pp-yoloe/paddlepaddle)                | PaddlePaddle          | COCO    |
| [PP-YOLOE+](cv/detection/pp_yoloe+/paddlepaddle)              | PaddlePaddle          | COCO    |
| [PVANet](cv/detection/pvanet/pytorch)                         | PyTorch               | COCO    |
| [RepPoints](cv/detection/reppoints_mmdet/pytorch)             | PyTorch (MMDetection) | COCO    |
| [RetinaNet](cv/detection/retinanet/pytorch)                   | PyTorch               | COCO    |
| [RetinaNet](cv/detection/retinanet/paddlepaddle)              | PaddlePaddle          | COCO    |
| [RT-DETR](cv/detection/rt-detr/pytorch)                       | PyTorch               | COCO    |
| [RTMDet](cv/detection/rtmdet/pytorch)                         | PyTorch               | COCO    |
| [SSD](cv/detection/ssd/pytorch)                               | PyTorch               | COCO    |
| [SSD](cv/detection/ssd/paddlepaddle)                          | PaddlePaddle          | COCO    |
| [SSD](cv/detection/ssd/tensorflow)                            | TensorFlow            | VOC     |
| [SSD](cv/detection/ssd/mindspore)                             | MindSpore             | COCO    |
| [YOLOF](cv/detection/yolof/pytorch)                           | PyTorch               | COCO    |
| [YOLOv3](cv/detection/yolov3/pytorch)                         | PyTorch               | COCO    |
| [YOLOv3](cv/detection/yolov3/paddlepaddle)                    | PaddlePaddle          | COCO    |
| [YOLOv3](cv/detection/yolov3/tensorflow)                      | TensorFlow            | VOC     |
| [YOLOv5](cv/detection/yolov5/paddlepaddle)                    | PaddlePaddle          | COCO    |
| [YOLOv5](cv/detection/yolov5/pytorch)                         | PyTorch               | COCO    |
| [YOLOv6](cv/detection/yolov6/pytorch)                         | PyTorch               | COCO    |
| [YOLOv7](cv/detection/yolov7/pytorch)                         | PyTorch               | COCO    |
| [YOLOv8](cv/detection/yolov8/pytorch)                         | PyTorch               | COCO    |
| [YOLOv9](cv/detection/yolov9/pytorch)                         | PyTorch               | COCO    |
| [YOLOv10](cv/detection/yolov10/pytorch)                       | PyTorch               | COCO    |

#### 3D Object Detection

| Model                                                         | Framework | Dataset          |
|---------------------------------------------------------------|-----------|------------------|
| [BEVFormer](cv/3d_detection/BEVFormer/pytorch)                | PyTorch   | nuScenes&CAN bus |
| [CenterPoint](cv/3d_detection/centerpoint/pytorch)            | PyTorch   | nuScenes         |
| [PAConv](cv/3d_detection/PAConv/pytorch)                      | PyTorch   | S3DIS            |
| [Part-A2-Anchor](cv/3d_detection/part_a2_anchor/pytorch)      | PyTorch   | KITTI            |
| [Part-A2-Free](cv/3d_detection/part_a2_free/pytorch)          | PyTorch   | KITTI            |
| [PointNet++](cv/3d_detection/pointnet2/pytorch/mmdetection3d) | PyTorch   | S3DIS            |
| [PointPillars](cv/3d_detection/pointpillars/pytorch)          | PyTorch   | KITTI            |
| [PointRCNN](cv/3d_detection/pointrcnn/pytorch)                | PyTorch   | KITTI            |
| [PointRCNN-IoU](cv/3d_detection/pointrcnn_iou/pytorch)        | PyTorch   | KITTI            |
| [SECOND](cv/3d_detection/second/pytorch)                      | PyTorch   | KITTI            |
| [SECOND-IoU](cv/3d_detection/second_iou/pytorch)              | PyTorch   | KITTI            |

#### 3D Reconstruction

| Model                                             | Framework | Dataset |
|---------------------------------------------------|-----------|---------|
| [HashNeRF](cv/3d-reconstruction/hashnerf/pytorch) | PyTorch   | fox     |

#### GNN (Graph Neural Network)

| Model                                      | Framework    | Dataset                  |
|--------------------------------------------|--------------|--------------------------|
| [GAT](cv/gnn/gat/paddlepaddle)             | PaddlePaddle | CORA                     |
| [GCN](cv/gnn/GCN/mindspore)                | MindSpore    | CORA & Citeseer          |
| [GCN](cv/gnn/GCN/paddlepaddle)             | PaddlePaddle | CORA & PubMed & Citeseer |
| [GraphSAGE](cv/gnn/graphsage/paddlepaddle) | PaddlePaddle | Reddit                   |

#### OCR

| Model                                          | Framework    | Dataset   |
|------------------------------------------------|--------------|-----------|
| [CRNN](cv/ocr/crnn/mindspore)                  | MindSpore    | OCR_Recog |
| [CRNN](cv/ocr/crnn/paddlepaddle)               | PaddlePaddle | LMDB      |
| [DBNet](cv/ocr/dbnet/pytorch)                  | PyTorch      | ICDAR2015 |
| [DBNet++](cv/ocr/dbnetpp/paddlepaddle)         | PaddlePaddle | ICDAR2015 |
| [DBNet++](cv/ocr/dbnetpp/pytorch)              | PyTorch      | ICDAR2015 |
| [PP-OCR-DB](cv/ocr/pp-ocr-db/paddlepaddle)     | PaddlePaddle | ICDAR2015 |
| [PP-OCR-EAST](cv/ocr/pp-ocr-east/paddlepaddle) | PaddlePaddle | ICDAR2015 |
| [PSE](cv/ocr/pse/paddlepaddle)                 | PaddlePaddle | OCR_Recog |
| [SAR](cv/ocr/sar/pytorch)                      | PyTorch      | OCR_Recog |
| [SAST](cv/ocr/sast/paddlepaddle)               | PaddlePaddle | ICDAR2015 |
| [SATRN](cv/ocr/satrn/pytorch/base)             | PyTorch      | OCR_Recog |

#### Point Cloud

| Model                                           | Framework | Dataset                         |
|-------------------------------------------------|-----------|---------------------------------|
| [Point-BERT](cv/point_cloud/point-bert/pytorch) | PyTorch   | ShapeNet55 & processed ModelNet |

#### Pose Estimation

| Model                                   | Framework    | Dataset |
|-----------------------------------------|--------------|---------|
| [AlphaPose](cv/pose/alphapose/pytorch)  | PyTorch      | COCO    |
| [HRNet](cv/pose/hrnet/pytorch)          | PyTorch      | COCO    |
| [HRNet-W32](cv/pose/hrnet/paddlepaddle) | PaddlePaddle | COCO    |
| [OpenPose](cv/pose/openpose/mindspore)  | MindSpore    | COCO    |

#### Self-Supervised Learning

| Model                                          | Framework | Dataset  |
|------------------------------------------------|-----------|----------|
| [MAE](cv/self_supervised_learning/MAE/pytorch) | PyTorch   | ImageNet |

#### Semantic Segmentation

| Model                                                                | Framework    | Dataset        |
|----------------------------------------------------------------------|--------------|----------------|
| [3D-UNet](cv/semantic_segmentation/unet3d/pytorch)                   | PyTorch      | kits19         |
| [APCNet](cv/semantic_segmentation/apcnet/pytorch)                    | PyTorch      | Cityscapes     |
| [Attention U-net](cv/semantic_segmentation/att_unet/pytorch)         | PyTorch      | Cityscapes     |
| [BiSeNet](cv/semantic_segmentation/bisenet/pytorch)                  | PyTorch      | COCO           |
| [BiSeNetV2](cv/semantic_segmentation/bisenetv2/paddlepaddle)         | PaddlePaddle | Cityscapes     |
| [BiSeNetV2](cv/semantic_segmentation/bisenetv2/pytorch)              | PyTorch      | Cityscapes     |
| [CGNet](cv/semantic_segmentation/cgnet/pytorch)                      | PyTorch      | COCO           |
| [ContextNet](cv/semantic_segmentation/contextnet/pytorch)            | PyTorch      | COCO           |
| [DabNet](cv/semantic_segmentation/dabnet/pytorch)                    | PyTorch      | COCO           |
| [DANet](cv/semantic_segmentation/danet/pytorch)                      | PyTorch      | COCO           |
| [DDRnet](cv/semantic_segmentation/ddrnet/pytorch)                    | PyTorch      | Cityscapes     |
| [DeepLabV3](cv/semantic_segmentation/deeplabv3/pytorch)              | PyTorch      | COCO           |
| [DeepLabV3](cv/semantic_segmentation/deeplabv3/paddlepaddle)         | PaddlePaddle | Cityscapes     |
| [DeepLabV3](cv/semantic_segmentation/deeplabv3/mindspore)            | MindSpore    | VOC            |
| [DeepLabV3+](cv/semantic_segmentation/deeplabv3plus/paddlepaddle)    | PaddlePaddle | Cityscapes     |
| [DeepLabV3+](cv/semantic_segmentation/deeplabv3plus/tensorflow)      | TensorFlow   | Cityscapes     |
| [DenseASPP](cv/semantic_segmentation/denseaspp/pytorch)              | PyTorch      | COCO           |
| [DFANet](cv/semantic_segmentation/dfanet/pytorch)                    | PyTorch      | COCO           |
| [DNLNet](cv/semantic_segmentation/dnlnet/paddlepaddle)               | PaddlePaddle | Cityscapes     |
| [DUNet](cv/semantic_segmentation/dunet/pytorch)                      | PyTorch      | COCO           |
| [EncNet](cv/semantic_segmentation/encnet/pytorch)                    | PyTorch      | COCO           |
| [ENet](cv/semantic_segmentation/enet/pytorch)                        | PyTorch      | COCO           |
| [ERFNet](cv/semantic_segmentation/erfnet/pytorch)                    | PyTorch      | COCO           |
| [ESPNet](cv/semantic_segmentation/espnet/pytorch)                    | PyTorch      | COCO           |
| [FastFCN](cv/semantic_segmentation/fastfcn/paddlepaddle)             | PyTorch      | ADE20K         |
| [FastSCNN](cv/semantic_segmentation/fastscnn/pytorch)                | PyTorch      | COCO           |
| [FCN](cv/semantic_segmentation/fcn/pytorch)                          | PyTorch      | COCO           |
| [FPENet](cv/semantic_segmentation/fpenet/pytorch)                    | PyTorch      | COCO           |
| [GCNet](cv/semantic_segmentation/gcnet/pytorch)                      | PyTorch      | Cityscapes     |
| [HardNet](cv/semantic_segmentation/hardnet/pytorch)                  | PyTorch      | COCO           |
| [ICNet](cv/semantic_segmentation/icnet/pytorch)                      | PyTorch      | COCO           |
| [LedNet](cv/semantic_segmentation/lednet/pytorch)                    | PyTorch      | COCO           |
| [LinkNet](cv/semantic_segmentation/linknet/pytorch)                  | PyTorch      | COCO           |
| [Mask2Former](cv/semantic_segmentation/Mask2Former/pytorch)          | PyTorch      | Cityscapes     |
| [MobileSeg](cv/semantic_segmentation/mobileseg/paddlepaddle)         | PaddlePaddle | Cityscapes     |
| [OCNet](cv/semantic_segmentation/ocnet/pytorch)                      | PyTorch      | COCO           |
| [OCRNet](cv/semantic_segmentation/ocrnet/paddlepaddle)               | PaddlePaddle | Cityscapes     |
| [OCRNet](cv/semantic_segmentation/ocrnet/pytorch)                    | PyTorch      | Cityscapes     |
| [PP-HumanSegV1](cv/semantic_segmentation/pp_humansegv1/paddlepaddle) | PaddlePaddle | PP-HumanSeg14K |
| [PP-HumanSegV2](cv/semantic_segmentation/pp_humansegv2/paddlepaddle) | PaddlePaddle | PP-HumanSeg14K |
| [PP-LiteSeg](cv/semantic_segmentation/pp_liteseg/paddlepaddle)       | PaddlePaddle | Cityscapes     |
| [PSANet](cv/semantic_segmentation/psanet/pytorch)                    | PyTorch      | COCO           |
| [RefineNet](cv/semantic_segmentation/refinenet/pytorch)              | PyTorch      | COCO           |
| [SegNet](cv/semantic_segmentation/segnet/pytorch)                    | PyTorch      | COCO           |
| [STDC](cv/semantic_segmentation/stdc/paddlepaddle)                   | PaddlePaddle | Cityscapes     |
| [STDC](cv/semantic_segmentation/stdc/pytorch)                        | PyTorch      | Cityscapes     |
| [UNet](cv/semantic_segmentation/unet/pytorch)                        | PyTorch      | COCO           |
| [UNet](cv/semantic_segmentation/unet/paddlepaddle)                   | PaddlePaddle | Cityscapes     |
| [UNet++](cv/semantic_segmentation/unet++/pytorch)                    | PyTorch      | DRIVE          |
| [VNet](cv/semantic_segmentation/vnet/tensorflow)                     | TensorFlow   | Hippocampus    |

#### Super Resolution

| Model                                                     | Framework | Dataset |
|-----------------------------------------------------------|-----------|---------|
| [basicVSR++](cv/super_resolution/basicVSR++/pytorch)      | PyTorch   | REDS    |
| [basicVSR](cv/super_resolution/basicVSR/pytorch)          | PyTorch   | REDS    |
| [ESRGAN](cv/super_resolution/esrgan/pytorch)              | PyTorch   | DIV2K   |
| [LIIF](cv/super_resolution/liif/pytorch)                  | PyTorch   | DIV2K   |
| [RealBasicVSR](cv/super_resolution/real_basicVSR/pytorch) | PyTorch   | REDS    |
| [TTSR](cv/super_resolution/ttsr/pytorch)                  | PyTorch   | CUFED   |
| [TTVSR](cv/super_resolution/ttvsr/pytorch)                | PyTorch   | REDS    |

#### Multi-Object Tracking

| Model                                           | Framework    | Dataset     |
|-------------------------------------------------|--------------|-------------|
| [ByteTrack](cv/tracking/bytetrack/paddlepaddle) | PaddlePaddle | MOT17       |
| [DeepSORT](cv/tracking/deep_sort/pytorch)       | PyTorch      | Market-1501 |
| [FairMOT](cv/tracking/fairmot/pytorch)          | PyTorch      | MOT17       |

### Multimodal

| Model                                                                  | Framework | Dataset        |
|------------------------------------------------------------------------|-----------|----------------|
| [BLIP](multimodal/BLIP/pytorch)                                        | PyTorch   | COCO           |
| [CLIP](multimodal/Language-Image_Pre-Training/clip/pytorch)            | PyTorch   | CIFAR100       |
| [ControlNet](multimodal/diffusion/ControlNet)                          | PyTorch   | Fill50K        |
| [DDPM](multimodal/diffusion/ddpm)                                      | PyTorch   | CIFAR-10       |
| [LLaVA 1.5](multimodal/llava/pytorch)                                  | PyTorch   | LLaVA-Pretrain |
| [L-Verse](multimodal/Language-Image_Pre-Training/L-Verse/pytorch)      | PyTorch   | ImageNet       |
| [Stable Diffusion 1.4](multimodal/diffusion/stable-diffusion/training) | PyTorch   | pokemon-images |
| [Stable Diffusion 1.5](multimodal/diffusion/stable-diffusion/sd_1.5)   | PyTorch   | pokemon-images |
| [Stable Diffusion 2.1](multimodal/diffusion/stable-diffusion/sd_2.1)   | PyTorch   | pokemon-images |
| [Stable Diffusion 3](multimodal/diffusion/stable-diffusion/sd_3)       | PyTorch   | dog-example    |
| [Stable Diffusion XL](multimodal/diffusion/stable-diffusion/sd_xl)     | PyTorch   | pokemon-images |

### NLP (Natural Language Processing)

#### Cloze Test

| Model                                                   | Framework | Dataset               |
|---------------------------------------------------------|-----------|-----------------------|
| [GLM](nlp/cloze_test/glm/pytorch/GLMForMultiTokenCloze) | PyTorch   | GLMForMultiTokenCloze |

#### Dialogue Generation

| Model                                      | Framework | Dataset |
|--------------------------------------------|-----------|---------|
| [CPM](nlp/dialogue_generation/cpm/pytorch) | PyTorch   | STC     |

#### Language Modeling

| Model                                                            | Framework         | Dataset            |
|------------------------------------------------------------------|-------------------|--------------------|
| [BART](nlp/language_model/bart_fairseq/pytorch)                  | PyTorch (Fairseq) | RTE                |
| [BERT NER](nlp/ner/bert/pytorch)                                 | PyTorch           | CoNLL-2003         |
| [BERT Pretraining](nlp/language_model/bert/pytorch)              | PyTorch           | MLCommon Wikipedia |
| [BERT Pretraining](nlp/language_model/bert/paddlepaddle)         | PaddlePaddle      | MNLI               |
| [BERT Pretraining](nlp/language_model/bert/tensorflow/base)      | TensorFlow        | MNLI               |
| [BERT Pretraining](nlp/language_model/bert/mindspore)            | MindSpore         | SQuAD              |
| [BERT Text Classification](nlp/text_classification/bert/pytorch) | PyTorch           | GLUE               |
| [BERT Text Summerization](nlp/text_summarisation/bert/pytorch)   | PyTorch           | cnn_dailymail      |
| [BERT Question Answering](nlp/question_answering/bert/pytorch)   | PyTorch           | SQuAD              |
| [GPT2-Medium-EN](nlp/llm/gpt2-medium-en/paddlepaddle)            | PaddlePaddle      | SST-2              |
| [RoBERTa](nlp/language_model/roberta_fairseq/pytorch)            | PyTorch (Fairseq) | RTE                |
| [XLNet](nlp/language_model/xlnet/paddlepaddle)                   | PaddlePaddle      | SST-2              |

#### Text Correction

| Model                                           | Framework    | Dataset |
|-------------------------------------------------|--------------|---------|
| [ERNIE](nlp/text_correction/ernie/paddlepaddle) | PaddlePaddle | corpus  |

#### Translation

| Model                                                          | Framework         | Dataset |
|----------------------------------------------------------------|-------------------|---------|
| [Convolutional](nlp/translation/convolutional_fairseq/pytorch) | PyTorch (Fairseq) | WMT14   |
| [T5](nlp/translation/t5/pytorch)                               | PyTorch           | WMT14   |
| [Transformer](nlp/translation/transformer/paddlepaddle)        | PaddlePaddle      | WMT14   |
| [Transformer](nlp/translation/transformer_fairseq/pytorch)     | PyTorch (Fairseq) | IWSLT14 |

### Reinforcement Learning

| Model                                                              | Framework    | Dataset     |
|--------------------------------------------------------------------|--------------|-------------|
| [DQN](reinforcement_learning/q-learning-networks/dqn/paddlepaddle) | PaddlePaddle | CartPole-v0 |

### Audio

#### Speech Recognition

| Model                                                                                   | Framework       | Dataset  |
|-----------------------------------------------------------------------------------------|-----------------|----------|
| [Conformer](audio/speech_recognition/conformer_wenet/pytorch)                           | PyTorch (WeNet) | AISHELL  |
| [Efficient Conformer v2](audio/speech_recognition/efficient_conformer_v2_wenet/pytorch) | PyTorch (WeNet) | AISHELL  |
| [PP-ASR-Conformer](audio/speech_recognition/conformer/paddlepaddle)                     | PaddlePaddle    | AISHELL  |
| [RNN-T](audio/speech_recognition/rnnt/pytorch)                                          | PyTorch         | LJSpeech |
| [Transformer](audio/speech_recognition/transformer_wenet/pytorch)                       | PyTorch (WeNet) | AISHELL  |
| [U2++ Conformer](audio/speech_recognition/u2++_conformer_wenet/pytorch)                 | PyTorch (WeNet) | AISHELL  |
| [Unified Conformer](audio/speech_recognition/unified_conformer_wenet/pytorch)           | PyTorch (WeNet) | AISHELL  |

#### Speech Synthesis

| Model                                                                 | Framework    | Dataset     |
|-----------------------------------------------------------------------|--------------|-------------|
| [PP-TTS-FastSpeech2](audio/speech_synthesis/fastspeech2/paddlepaddle) | PaddlePaddle | CSMSC       |
| [PP-TTS-HiFiGAN](audio/speech_synthesis/hifigan/paddlepaddle)         | PaddlePaddle | CSMSC       |
| [Tacotron2](audio/speech_synthesis/tacotron2/pytorch)                 | PyTorch      | LJSpeech    |
| [VQMIVC](audio/speech_synthesis/vqmivc/pytorch)                       | PyTorch      | VCTK-Corpus |
| [WaveGlow](audio/speech_synthesis/waveglow/pytorch)                   | PyTorch      | LJSpeech    |

### Others

#### Graph Machine Learning

| Model                                                                | Framework | Dataset            |
|----------------------------------------------------------------------|-----------|--------------------|
| [Graph WaveNet](others/graph_machine_learning/graph_wavenet/pytorch) | PyTorch   | METR-LA & PEMS-BAY |

#### Kolmogorov-Arnold Networks

| Model                                                | Framework | Dataset |
|------------------------------------------------------|-----------|---------|
| [KAN](others/kolmogorov_arnold_networks/kan/pytorch) | PyTorch   | -       |

#### Model Pruning

| Model                                                             | Framework | Dataset      |
|-------------------------------------------------------------------|-----------|--------------|
| [Network Slimming](others/model_pruning/network-slimming/pytorch) | PyTorch   | CIFAR-10/100 |

#### Recommendation Systems

| Model                                                             | Framework    | Dataset         |
|-------------------------------------------------------------------|--------------|-----------------|
| [DeepFM](others/recommendation_systems/deepfm/paddlepaddle)       | PaddlePaddle | Criteo_Terabyte |
| [DLRM](others/recommendation_systems/dlrm/pytorch)                | PyTorch      | Criteo_Terabyte |
| [DLRM](others/recommendation_systems/dlrm/paddlepaddle)           | PaddlePaddle | Criteo_Terabyte |
| [FFM](others/recommendation_systems/ffm/paddlepaddle)             | PaddlePaddle | Criteo_Terabyte |
| [NCF](others/recommendation_systems/ncf/pytorch)                  | PyTorch      | movielens       |
| [Wide&Deep](others/recommendation_systems/wide_deep/paddlepaddle) | PaddlePaddle | Criteo_Terabyte |
| [xDeepFM](others/recommendation_systems/xdeepfm/paddlepaddle)     | PaddlePaddle | Criteo_Terabyte |

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
