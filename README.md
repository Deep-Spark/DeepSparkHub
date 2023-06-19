# DeepSparkHub

DeepSparkHub甄选上百个应用算法和模型，覆盖AI和通用计算各领域，支持主流市场智能计算场景，包括智慧城市、数字个人、医疗、教育、通信、能源等多个领域。

## 模型列表

注：当前列表中模型主要基于PyTorch框架，基于PaddlePaddle、TensorFlow、MindSpore等框架的模型在逐步添加中。

- Computer Vision

   - [Classification](#classification)
   - [Face Detection](#face-detection)
   - [Face Recognition](#face-recognition)
   - [Instance Segmentation](#instance-segmentation)
   - [Knowledge Distillation](#knowledge-distillation)
   - [Network Pruning](#network-pruning)
   - [Object Detection](#object-detection)
   - [3D Object Detection](#3d-object-detection)
   - [OCR](#ocr)
   - [Point Cloud](#point-cloud)
   - [Pose Estimation](#pose-estimation)
   - [Self-Supervised Learning](#self-supervised-learning)
   - [Semantic Segmentation](#semantic-segmentation)
   - [Super Resolution](#super-resolution)
   - [Tracking](#tracking)
   - [Traffic Forecast](#traffic-forecast)

- [Multimodal](#multimodal)

- [GNN](#gnn)

- Natural Language Processing(NLP)

   - [Cloze Test](#cloze-test)
   - [Dialogue Generation](#dialogue-generation)
   - [Language Modeling](#language-modeling)
   - [Text Correction](#text-correction)
   - [Translation](#translation)

- Recommendation

   - [Collaborative Filtering](#collaborative-filtering)
   - [CTR](#ctr)

- Speech

   - [Speech Recognition](#speech-recognition)
   - [Speech Synthesis](#speech-synthesis)

- [3D Reconstruction](#3d-reconstruction)

  
--------

### Computer Vision

#### Classification

模型名称 | 框架 | 数据集 
-------- | ------ | ---- 
[ACmix](cv/classification/acmix/pytorch/README.md)  | PyTorch |  ImageNet
[AlexNet](cv/classification/alexnet/pytorch/README.md)  | PyTorch | ImageNet
[CBAM](cv/classification/cbam/pytorch/README.md)  | PyTorch | ImageNet
[ConvNext](cv/classification/convnext/pytorch/README.md)  | PyTorch | ImageNet
[CspDarknet53](cv/classification/cspdarknet53/pytorch/README.md)  | PyTorch | ImageNet
[DenseNet](cv/classification/densenet/pytorch/README.md)  | PyTorch | ImageNet 
[DPN92](cv/classification/dpn92/pytorch/README.md)  | PyTorch | ImageNet
[DPN107](cv/classification/dpn107/pytorch/README.md)  | PyTorch | ImageNet
[ECA_MobileNet_V2](cv/classification/eca_mobilenet_v2/pytorch/README.md)  | PyTorch | ImageNet
[ECA_RESNET152](cv/classification/eca_resnet152/pytorch/README.md)  | PyTorch | ImageNet
[Efficientb4](cv/classification/efficientb4/pytorch/README.md)  | PyTorch | ImageNet
[FasterNet](cv/classification/fasternet/pytorch/README.md)  | PyTorch | ImageNet
[googlenet](cv/classification/googlenet/pytorch/README.md)  | PyTorch | ImageNet
[googlenet](cv/classification/googlenet/paddlepaddle/README.md)  | PaddlePaddle | ImageNet
[InceptionV3](cv/classification/inceptionv3/pytorch/README.md)  | PyTorch | ImageNet
[InceptionV4](cv/classification/inceptionv4/pytorch/README.md)  | PyTorch | ImageNet
[lenet](cv/classification/lenet/pytorch/README.md)  | PyTorch | ImageNet
[MobileNetV2](cv/classification/mobilenetv2/pytorch/README.md)  | PyTorch | ImageNet
[MobileNetV3](cv/classification/mobilenetv3/pytorch/README.md)  | PyTorch | ImageNet
[MobileNetV3](cv/classification/mobilenetv3/paddlepaddle/README.md)  | PaddlePaddle | ImageNet
[RepVGG](cv/classification/repvgg/pytorch/README.md)  | PyTorch | ImageNet
[RepVGG](cv/classification/repvgg/paddlepaddle/README.md)  | PaddlePaddle | ImageNet
[ResNeSt14](cv/classification/resnest14/pytorch/README.md)  | PyTorch | ImageNet
[ResNeSt50](cv/classification/resnest50/pytorch/README.md)  | PyTorch | ImageNet
[ResNeSt50](cv/classification/resnest50/paddlepaddle/README.md)  | PaddlePaddle | ImageNet
[ResNeSt101](cv/classification/resnest101/pytorch/README.md)  | PyTorch | ImageNet
[ResNeSt269](cv/classification/resnest269/pytorch/README.md)  | PyTorch | ImageNet
[ResNet18](cv/classification/resnet18/pytorch/README.md)  | PyTorch | ImageNet
[ResNet50](cv/classification/resnet50/pytorch/README.md)  | PyTorch | ImageNet
[ResNet50](cv/classification/resnet50/paddlepaddle/README.md)  | PaddlePaddle | ImageNet
[ResNet50](cv/classification/resnet50/tensorflow/README.md)  | TensorFlow | ImageNet
[ResNet101](cv/classification/resnet101/pytorch/README.md)  | PyTorch | ImageNet
[ResNet152](cv/classification/resnet152/pytorch/README.md)  | PyTorch | ImageNet
[ResNeXt50_32x4d](cv/classification/resnext50_32x4d/pytorch/README.md)  | PyTorch | ImageNet
[ResNeXt101_32x8d](cv/classification/resnext101_32x8d/pytorch/README.md)  | PyTorch | ImageNet
[SEResNeXt](cv/classification/seresnext/pytorch/README.md)  | PyTorch | ImageNet
[ShuffleNetV2](cv/classification/shufflenetv2/pytorch/README.md)  | PyTorch | ImageNet
[SqueezeNet](cv/classification/squeezenet/pytorch/README.md)  | PyTorch | ImageNet
[Swin Transformer](cv/classification/swin_transformer/pytorch/README.md)  | PyTorch | ImageNet
[Swin Transformer](cv/classification/swin_transformer/paddlepaddle/README.md)  | PaddlePaddle | ImageNet
[VGG16](cv/classification/vgg/pytorch/README.md)  | PyTorch | ImageNet
[VGG16](cv/classification/vgg/paddlepaddle/README.md)  | PaddlePaddle | ImageNet
[Wave-MLP](cv/classification/wavemlp/pytorch/README.md)  | PyTorch | ImageNet
[Wide_ResNet101_2](cv/classification/wide_resnet101_2/pytorch/README.md)  | PyTorch | ImageNet
[Xception](cv/classification/xception/pytorch/README.md)  | PyTorch | ImageNet

#### Face Detection

模型名称 | 框架 | 数据集 
-------- | ------ | ---- 
[RetinaFace](cv/face/retinaface/pytorch/README.md)  | PyTorch | WiderFace

#### Face Recognition

模型名称 | 框架 | 数据集 
-------- | ------ | ---- 
[ArcFace](cv/face/arcface/pytorch/README.md)  | PyTorch | CASIA-WebFaces&LFW
[CosFace](cv/face/cosface/pytorch/README.md)  | PyTorch | CASIA-WebFaces&LFW
[FaceNet](cv/face/facenet/pytorch/README.md)  | PyTorch | CASIA-WebFaces&LFW

#### Instance Segmentation

模型名称 | 框架 | 数据集 
-------- | ------ | ---- 
[SOLO](cv/instance_segmentation/SOLO/pytorch/README.md)  | PyTorch | COCO
[SOLOv2](cv/detection/solov2/paddlepaddle/README.md)  | PaddlePaddle | COCO
[YOLACT++](cv/instance_segmentation/yolact/pytorch/README.md)  | PyTorch | COCO

#### Image Generation

模型名称 | 框架 | 数据集 
-------- | ------ | ---- 
[DCGAN](cv/image_generation/dcgan/MindSpore/README.md)  | MindSpore | ImageNet

#### Knowledge Distillation

模型名称 | 框架 | 数据集 
-------- | ------ | ---- 
[CWD](cv/distiller/CWD/pytorch/README.md)  | PyTorch | Cityscapes
[RKD](cv/distiller/RKD/pytorch/README.md)  | PyTorch | CUB-200-2011

#### Network Pruning

模型名称 | 框架 | 数据集 
-------- | ------ | ---- 
[Network Slimming](cv/Pruning/Network-Slimming/pytorch/README.md)  | PyTorch (Fairseq) | CIFAR-10/100

#### Object Detection

模型名称 | 框架 | 数据集 
-------- | ------ | ---- 
[ATSS](cv/detection/mmdetection_tools/ATSS/README.md)  | PyTorch (MMDetection) | COCO
[AutoAssign](cv/detection/autoassign/pytorch/README.md)  | PyTorch | COCO
[Cascade R-CNN](cv/detection/mmdetection_tools/cascade_rcnn/READNE.md)  | PyTorch (MMDetection) | COCO
[CenterNet](cv/detection/centernet/pytorch/README.md)  | PyTorch | COCO
[CenterNet](cv/detection/centernet/paddlepaddle/README.md)  | PaddlePaddle | COCO
[CornerNet](cv/detection/mmdetection_tools/CornerNet/README.md)  | PyTorch (MMDetection) | COCO
[DCNV2](cv/detection/mmdetection_tools/DCNV2/README.md)  | PyTorch (MMDetection) | COCO
[DeepSORT](cv/tracking/deep_sort/pytorch/README.md)  | PyTorch | Market-1501
[DETR](cv/detection/detr/paddlepaddle/README.md)  | PaddlePaddle | COCO
[Faster R-CNN](cv/detection/fasterrcnn/pytorch/README.md)  | PyTorch | COCO
[FCOS](cv/detection/fcos/paddlepaddle/README.md)  | PaddlePaddle | COCO
[FCOS](cv/detection/fcos/pytorch/README.md)  | PyTorch | COCO
[Mask R-CNN](cv/detection/maskrcnn/pytorch/README.md)  | PyTorch | COCO
[Mask R-CNN](cv/detection/maskrcnn/paddlepaddle/README.md)  | PaddlePaddle | COCO
[PP-YOLOE](cv/detection/pp-yoloe/paddlepaddle/README.md)  | PaddlePaddle | COCO
[PVANet](cv/detection/pvanet/pytorch/README.md)  | PyTorch | COCO
[RepPoints](cv/detection/mmdetection_tools/reppoints/README.md)  | PyTorch (MMDetection) | COCO
[RetinaNet](cv/detection/retinanet/pytorch/README.md)  | PyTorch | COCO
[RetinaNet](cv/detection/retinanet/paddlepaddle/README.md)  | PaddlePaddle | COCO
[SSD](cv/detection/ssd/pytorch/README.md)  | PyTorch | COCO
[SSD](cv/detection/ssd/paddlepaddle/README.md)  | PaddlePaddle | COCO
[SSD](cv/detection/ssd/tensorflow/README.md)  | TensorFlow | VOC
[SSD](cv/detection/ssd/MindSpore/README.md)  | MindSpore | COCO
[YOLOF](cv/detection/yolof/pytorch/README.md)  | PyTorch | COCO
[YOLOv3](cv/detection/yolov3/pytorch/README.md)  | PyTorch | COCO
[YOLOv3](cv/detection/yolov3/paddlepaddle/README.md)  | PaddlePaddle | COCO
[YOLOv3](cv/detection/yolov3/tensorflow/README.md)  | TensorFlow | VOC
[YOLOv5](cv/detection/yolov5/pytorch/README.md)  | PyTorch | COCO
[YOLOv6](cv/detection/yolov6/pytorch/README.md)  | PyTorch | COCO
[YOLOv7](cv/detection/yolov7/pytorch/README.md)  | PyTorch | COCO
[YOLOv8](cv/detection/yolov8/pytorch/README.md)  | PyTorch | COCO

#### 3D Object Detection

模型名称 | 框架 | 数据集 
-------- | ------ | ---- 
[BEVFormer](cv/3d_detection/BEVFormer/pytorch/README.md)  | PyTorch | nuScenes&CAN bus
[PointNet++](cv/3d_detection/pointnet2/pytorch/mmdetection3d/README.md)  | PyTorch | S3DIS
[PointPillars](cv/3d_detection/pointpillars/pytorch/README.md)  | PyTorch | KITTI

#### OCR

模型名称 | 框架 | 数据集 
-------- | ------ | ---- 
[CRNN](cv/ocr/crnn/paddlepaddle/README.md)  | PaddlePaddle | LMDB
[DBNet](cv/ocr/dbnet/pytorch/README.md)  | PyTorch | ICDAR2015
[PP-OCR-DB](cv/ocr/pp-ocr-db/paddlepaddle/README.md)  | PaddlePaddle | ICDAR2015
[PSE](cv/ocr/pse/paddlepaddle/README.md)  | PaddlePaddle | OCR_Recog
[SAR](cv/ocr/sar/pytorch/README.md)  | PyTorch | OCR_Recog
[SATRN](cv/ocr/satrn/pytorch/base/README.md)  | PyTorch | OCR_Recog

#### Point Cloud

模型名称 | 框架 | 数据集 
-------- | ------ | ---- 
[Point-BERT](cv/point_cloud/Point-BERT/pytorch/README.md)  | PyTorch | ShapeNet55 & processed ModelNet

#### Pose Estimation

模型名称 | 框架 | 数据集 
-------- | ------ | ---- 
[AlphaPose](cv/pose/alphapose/pytorch/README.md)  | PyTorch | COCO
[HRNet](cv/pose/hrnet/pytorch/README.md)  | PyTorch | COCO

#### Self-Supervised Learning

模型名称 | 框架 | 数据集 
-------- | ------ | ---- 
[MAE](cv/self_supervised_learning/MAE/pytorch/README.md)  | PyTorch | ImageNet

#### Semantic Segmentation

模型名称 | 框架 | 数据集 
-------- | ------ | ---- 
[3D-UNet](cv/semantic_segmentation/unet3d/pytorch/README.md)  | PyTorch | kits19
[APCNet](cv/semantic_segmentation/apcnet/pytorch/README.md)  | PyTorch | Cityscapes
[Attention U-net](cv/semantic_segmentation/att_unet/pytorch/README.md)  | PyTorch | Cityscapes
[BiSeNet](cv/semantic_segmentation/bisenet/pytorch/README.md)  | PyTorch | COCO
[BiSeNetV2](cv/semantic_segmentation/bisenetv2/paddlepaddle/README.md)  | PaddlePaddle | Cityscapes
[CGNet](cv/semantic_segmentation/cgnet/pytorch/README.md)  | PyTorch | COCO
[ContextNet](cv/semantic_segmentation/contextnet/pytorch/README.md)  | PyTorch | COCO
[DabNet](cv/semantic_segmentation/dabnet/pytorch/README.md)  | PyTorch | COCO
[DANet](cv/semantic_segmentation/danet/pytorch/README.md)  | PyTorch | COCO
[DDRnet](cv/semantic_segmentation/ddrnet/pytorch/README.md)  | PyTorch | Cityscapes
[DeepLabV3](cv/semantic_segmentation/deeplabv3/pytorch/README.md)  | PyTorch | COCO
[DeepLabV3](cv/semantic_segmentation/deeplabv3/paddlepaddle/README.md)  | PaddlePaddle | Cityscapes
[DeepLabV3](cv/semantic_segmentation/deeplabv3/MindSpore/README.md)  | MindSpore | VOC
[DeepLabV3+](cv/semantic_segmentation/deeplabv3plus/paddlepaddle/README.md)  | PaddlePaddle | Cityscapes
[DenseASPP](cv/semantic_segmentation/denseaspp/pytorch/README.md)  | PyTorch | COCO
[DFANet](cv/semantic_segmentation/dfanet/pytorch/README.md)  | PyTorch | COCO
[dnlnet](cv/semantic_segmentation/dnlnet/paddlepaddle/README.md)  | PaddlePaddle | Cityscapes
[DUNet](cv/semantic_segmentation/dunet/pytorch/README.md)  | PyTorch | COCO
[EncNet](cv/semantic_segmentation/encnet/pytorch/README.md)  | PyTorch | COCO
[ENet](cv/semantic_segmentation/enet/pytorch/README.md)  | PyTorch | COCO
[ERFNet](cv/semantic_segmentation/erfnet/pytorch/README.md)  | PyTorch | COCO
[ESPNet](cv/semantic_segmentation/espnet/pytorch/README.md)  | PyTorch | COCO
[FastSCNN](cv/semantic_segmentation/fastscnn/pytorch/README.md)  | PyTorch | COCO
[FCN](cv/semantic_segmentation/fcn/pytorch/README.md)  | PyTorch | COCO
[FPENet](cv/semantic_segmentation/fpenet/pytorch/README.md)  | PyTorch | COCO
[GCNet](cv/semantic_segmentation/gcnet/pytorch/README.md)  | PyTorch | Cityscapes
[HardNet](cv/semantic_segmentation/hardnet/pytorch/README.md)  | PyTorch | COCO
[ICNet](cv/semantic_segmentation/icnet/pytorch/README.md)  | PyTorch | COCO
[LedNet](cv/semantic_segmentation/lednet/pytorch/README.md)  | PyTorch | COCO
[LinkNet](cv/semantic_segmentation/linknet/pytorch/README.md)  | PyTorch | COCO
[OCNet](cv/semantic_segmentation/ocnet/pytorch/README.md)  | PyTorch | COCO
[OCRNet](cv/semantic_segmentation/ocrnet/pytorch/README.md)  | PyTorch | Cityscapes
[PSANet](cv/semantic_segmentation/psanet/pytorch/README.md)  | PyTorch | COCO
[RefineNet](cv/semantic_segmentation/refinenet/pytorch/README.md)  | PyTorch | COCO
[SegNet](cv/semantic_segmentation/segnet/pytorch/README.md)  | PyTorch | COCO
[STDC](cv/semantic_segmentation/stdc/pytorch/README.md)  | PyTorch | Cityscapes
[UNet](cv/semantic_segmentation/unet/pytorch/README.md)  | PyTorch | COCO
[UNet](cv/semantic_segmentation/unet/paddlepaddle/README.md)  | PaddlePaddle | Cityscapes
[UNet++](cv/semantic_segmentation/unet++/pytorch/README.md)  | PyTorch | DRIVE
[VNet](cv/semantic_segmentation/vnet/tensorflow/README.md)  | TensorFlow | Hippocampus

#### Super Resolution

模型名称 | 框架 | 数据集 
-------- | ------ | ---- 
[basicVSR++](cv/super_resolution/basicVSR++/pytorch/README.md)  | PyTorch | REDS
[basicVSR](cv/super_resolution/basicVSR/pytorch/README.md)  | PyTorch | REDS
[ESRGAN](cv/super_resolution/esrgan/pytorch/README.md)  | PyTorch | DIV2K 
[LIIF](cv/super_resolution/liif/pytorch/README.md)  | PyTorch | DIV2K
[RealBasicVSR](cv/super_resolution/real_basicVSR/pytorch/README.md)  | PyTorch | REDS
[TTSR](cv/super_resolution/ttsr/pytorch/README.md)  | PyTorch | CUFED
[TTVSR](cv/super_resolution/ttvsr/pytorch/README.md)  | PyTorch | REDS

#### Tracking

模型名称 | 框架 | 数据集 
-------- | ------ | ---- 
[FairMOT](cv/tracking/fairmot/pytorch/README.md)  | PyTorch | MOT17

#### Traffic Forecast

模型名称 | 框架 | 数据集 
-------- | ------ | ---- 
[Graph WaveNet](cv/traffic_forecast/graph_wavenet/pytorch/README.md)  | PyTorch | METR-LA & PEMS-BAY

### Multimodal

模型名称 | 框架 | 数据集 
-------- | ------ | ---- 
[CLIP](multimodal/Language-Image_Pre-Training/clip/pytorch/README.md)  | PyTorch | CIFAR100
[L-Verse](multimodal/Language-Image_Pre-Training/L-Verse/pytorch/README.md)  | PyTorch | ImageNet
[Stable Diffusion](multimodal/diffusion/stable-diffusion/training/README.md)  | PyTorch | pokemon-images

### GNN

#### Text Classification
模型名称 | 框架 | 数据集 
-------- | ------ | ---- 
[GCN](gnn/GCN/README.md)  | MindSpore | Cora&Citeseer

### NLP

#### Cloze Test

模型名称 | 框架 | 数据集 
-------- | ------ | ---- 
[GLM](nlp/cloze_test/glm/pytorch/GLMForMultiTokenCloze/README.md)  | PyTorch | GLMForMultiTokenCloze

#### Dialogue Generation

模型名称 | 框架 | 数据集 
-------- | ------ | ---- 
[CPM](nlp/dialogue_generation/cpm/pytorch/README.md)  | PyTorch | STC

#### Language Modeling

模型名称 | 框架 | 数据集 
-------- | ------ | ---- 
[BART](fairseq-tools/README_bart.md)  | PyTorch (Fairseq) | RTE
[BERT NER](nlp/ner/bert/pytorch/README.md) | PyTorch | CoNLL-2003
[BERT Pretraining](nlp/language_model/bert/pytorch/README.md)  | PyTorch | MLCommon Wikipedia (2048_shards_uncompressed)
[BERT Pretraining](nlp/language_model/bert/paddlepaddle/README.md)  | PaddlePaddle | MNLI
[BERT Pretraining](nlp/language_model/bert/tensorflow/base/README.md)  | TensorFlow | MNLI
[BERT Pretraining](nlp/language_model/bert/MindSpore/README.md)  | MindSpore | SQuAD
[BERT Text Classification](nlp/text_classification/bert/pytorch/README.md) |PyTorch | GLUE
[BERT Text Summerization](nlp/text_summarisation/bert/pytorch/README.md) | PyTorch | cnn_dailymail
[BERT Question Answering](nlp/question_answering/bert/pytorch/README.md) | PyTorch | SQuAD
[RoBERTa](fairseq-tools/README_roberta.md)  | PyTorch (Fairseq) | RTE

#### Text Correction
模型名称 | 框架 | 数据集 
-------- | ------ | ---- 
[Ernie](nlp/text_correction/ernie/paddlepaddle/README.md)  | PaddlePaddle | corpus

#### Translation

模型名称 | 框架 | 数据集 
-------- | ------ | ---- 
[Convolutional](fairseq-tools/README_convolutional.md)  | PyTorch (Fairseq) | WMT14
[T5](nlp/translation/t5/pytorch/README.md)  | PyTorch | wmt14-en-de-pre-processed
[Transformer](nlp/translation/transformer/paddlepaddle/README.md)  | PaddlePaddle | wmt14-en-de-pre-processed
[Transformer](fairseq-tools/README_transformer.md)  | PyTorch (Fairseq) | IWSLT14

### Recommendation

#### Collaborative Filtering

模型名称 | 框架 | 数据集 
-------- | ------ | ---- 
[NCF](recommendation/collaborative_filtering/ncf/pytorch/README.md)  | PyTorch | movielens

#### CTR

模型名称 | 框架 | 数据集 
-------- | ------ | ---- 
[DLRM](recommendation/ctr/dlrm/pytorch/README.md)  | PyTorch | Criteo_Terabyte
[DeepFM](recommendation/deepfm/paddlepaddle/README.md)  | PaddlePaddle | Criteo_Terabyte
[Wide&Deep](recommendation/wide_deep/paddlepaddle/README.md)  | PaddlePaddle | Criteo_Terabyte

### Speech

#### Speech Recognition

模型名称 | 框架 | 数据集 
-------- | ------ | ---- 
[Conformer](speech/speech_recognition/conformer/pytorch/README.md)  | PyTorch | AISHELL
[RNN-T](speech/speech_recognition/rnnt/pytorch/README.md)  | PyTorch | LJSpeech
[Transformer](speech/speech_recognition/transformer/pytorch/README.md)  | PyTorch (WeNet) | AISHELL
[U2++ Conformer](speech/speech_recognition/u2++_conformer/pytorch/README.md)  | PyTorch (WeNet) | AISHELL
[Unified Conformer](speech/speech_recognition/unified_conformer/pytorch/README.md)  | PyTorch (WeNet) | AISHELL

#### Speech Synthesis

模型名称 | 框架 | 数据集 
-------- | ------ | ---- 
[Tacotron2](speech/speech_synthesis/tacotron2/pytorch/README.md)  | PyTorch | LJSpeech
[vqmivc](speech/speech_synthesis/vqmivc/pytorch/README.md)  | PyTorch | VCTK-Corpus
[waveglow](speech/speech_synthesis/waveglow/pytorch/README.md)  | PyTorch | LJSpeech

### 3D Reconstruction

模型名称 | 框架 | 数据集 
-------- | ------ | ---- 
[ngp-nerf](3d-reconstruction/hashnerf/readme.md)  | PyTorch | fox

-------

## 容器镜像构建方式

社区用户可参考[容器镜像构建说明](docker/Iluvatar/README.md)在本地构建出能够运行DeepSparkHub仓库中模型的容器镜像。

-------

## 社区

### 治理

请参见 DeepSpark Code of Conduct on [Gitee](https://gitee.com/deep-spark/deepspark/blob/master/CODE_OF_CONDUCT.md) or on [GitHub](https://github.com/Deep-Spark/deepspark/blob/main/CODE_OF_CONDUCT.md)。

### 交流

请联系 contact@deepspark.org.cn。

### 贡献

请参见 [DeepSparkHub Contributing Guidelines](CONTRIBUTING.md)。

## 许可证

本项目许可证遵循[Apache-2.0](LICENSE)。

