<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable line-length -->
[<img alt="English" src="https://img.shields.io/badge/Language-English-blue.svg">](README_en.md) [<img alt="Chinese" src="https://img.shields.io/badge/语言-简体中文-red.svg">](README.md)

# DeepSparkHub

<div align="center" style="line-height: 1;">
  <a href="https://www.deepspark.org.cn"><img alt="Homepage"
    src="https://img.shields.io/badge/DeepSpark-Homepage-blue.svg"/></a>
  <a href="./LICENSE"><img alt="LICENSE" src="https://img.shields.io/badge/license-Apache%202.0-dfd.svg"></a>
  <a href="https://gitee.com/deep-spark/deepsparkhub/releases/latest"><img alt="Release" src="https://img.shields.io/github/v/release/deep-spark/deepsparkhub?color=ffa"></a>
</div>
<br>

DeepSparkHub selects more than 300 application algorithms and models, covering various fields of AI and general
computing. It supports mainstream intelligent computing scenarios in the market, including smart cities, digital
individuals, healthcare, education, communication, energy, and more.

## ModelZoo

### LLM (Large Language Model)

| Model                                                 | Framework            | Dataset/Weight         | IXUCA SDK |
|-------------------------------------------------------|--------------------|------------------------|-----------|
| [Aquila2-34B](models/nlp/llm/aquila2-34b/pytorch)            | Megatron-DeepSpeed | Bookcorpus             | 3.4.0     |
| [Baichuan2-7B](models/nlp/llm/baichuan2-7b/pytorch)          | DeepSpeed          | baichuan2-7b-base      | 3.4.0     |
| [Bloom-7B1](models/nlp/llm/bloom-7b1/pytorch)                | Firefly            | school_math_0.25M      | 3.4.0     |
| [ChatGLM-6B](models/nlp/llm/chatglm-6b/pytorch)              | DeepSpeed          | ADGEN & chatglm-6b     | 3.1.0     |
| [ChatGLM2-6B SFT](models/nlp/llm/chatglm2-6b-sft/pytorch)    | DeepSpeed          | ADGEN & chatglm2-6b    | 3.4.0     |
| [ChatGLM3-6B](models/nlp/llm/chatglm3-6b/pytorch)            | DeepSpeed          | ADGEN & chatglm3-6b    | 4.1.1     |
| [DeepSeekMoE 7B](models/nlp/llm/deepseek_moe_7b/pytorch)     | ColossalAI         | deepseek-moe-16b-base  | 4.1.1     |
| [DeepSeek-LLM-7B](models/nlp/llm/deepseek-llm-7b/verl)       | verl               | deepseek-llm-7b-chat   | dev-only  |
| [GLM-4](models/nlp/llm/glm-4/pytorch)                        | Torchrun           | glm-4-9b-chat          | 4.2.0     |
| [Gemma-2-2B-IT](models/nlp/llm/gemma-2-2b-it/verl)           | verl               | gemma-2-2b-it          | dev-only  |
| [Llama-7B](models/nlp/llm/llama-7b/pytorch)                  | ColossalAI         | llama-7b-hf            | 3.1.0     |
| [Llama2-7B](models/nlp/llm/llama2-7b/pytorch)                | Megatron-DeepSpeed | Bookcorpus             | 3.1.0     |
| [Llama2-7B RMF](models/nlp/llm/llama2-7b_reward_sft/pytorch) | DeepSpeed          | Dahoas/rm-static       | 3.1.1     |
| [Llama2-7B RLHF](models/nlp/llm/llama2-7b_rlhf/pytorch)      | Megatron-DeepSpeed | llama2-7b&tiny-llama   | 3.4.0     |
| [Llama2-7B SFT](models/nlp/llm/llama2-7b_sft/pytorch)        | Megatron-DeepSpeed | GPT Small-117M         | 3.1.1     |
| [Llama2-13B](models/nlp/llm/llama2-13b/pytorch)              | Megatron-DeepSpeed | Bookcorpus             | 3.4.0     |
| [Llama2-34B](models/nlp/llm/llama2-34b/pytorch)              | Megatron-DeepSpeed | Bookcorpus             | 3.4.0     |
| [Llama3-8B](models/nlp/llm/llama3_8b/pytorch)                | Megatron-DeepSpeed | Bookcorpus             | 4.1.1     |
| [Llama3-8B](models/nlp/llm/llama3_8b/megatron-lm)            | Megatron-LM        | GPT Small-117M         | 4.3.0     |
| [Llama3-8B SFT](models/nlp/llm/llama3_8b_sft/pytorch)        | ColossalAI         | school_math_0.25M      | 4.1.1     |
| [Llama3-8B SFT](models/nlp/llm/llama3_8b/openrlhf)           | OpenRLHF           | Meta-Llama-3-8B        | 4.3.0     |
| [Llama3-8B PPO](models/nlp/llm/llama3_8b/openrlhf)           | OpenRLHF           | Llama-3-8b-sft-mixture | 4.2.0     |
| [Llama3-8B DPO](models/nlp/llm/llama3_8b/openrlhf)           | OpenRLHF           | Llama-3-8b-sft-mixture | 4.2.0     |
| [Llama3-8B KTO](models/nlp/llm/llama3_8b/openrlhf)           | OpenRLHF           | Llama-3-8b-sft-mixture | 4.2.0     |
| [Llama3-8B DPO](models/nlp/llm/llama3_8b/llamafactory)       | LLaMA-Factory   | Meta-Llama-3-8B-Instruct | 4.3.0   |
| [Llama3-8B Full SFT](models/nlp/llm/llama3_8b/llamafactory)  | LLaMA-Factory   | Meta-Llama-3-8B-Instruct | 4.3.0   |
| [Llama3-8B KTO](models/nlp/llm/llama3_8b/llamafactory)       | LLaMA-Factory   | Meta-Llama-3-8B-Instruct | 4.3.0   |
| [Llama3-8B Pretrain](models/nlp/llm/llama3_8b/llamafactory)  | LLaMA-Factory   | Meta-Llama-3-8B-Instruct | 4.3.0   |
| [Llama3-8B Reward](models/nlp/llm/llama3_8b/llamafactory)    | LLaMA-Factory   | Meta-Llama-3-8B-Instruct | 4.3.0   |
| [Llama3-8B SFT](models/nlp/llm/llama3_8b/llamafactory)       | LLaMA-Factory   | Meta-Llama-3-8B-Instruct | 4.3.0   |
| [Mamba-2](models/nlp/llm/mamba-2/pytorch)                    | Megatron-LM        | GPT Small-117M         | 4.1.1     |
| [MiniCPM](models/nlp/llm/minicpm/pytorch)                    | DeepSpeed          | MiniCPM-2B-sft-bf16    | 4.2.0     |
| [Mixtral 8x7B](models/nlp/llm/mixtral/pytorch)               | Megatron-LM        | GPT Small-117M         | 4.1.1     |
| [Mixtral 8x7B](models/nlp/llm/mixtral/openrlhf)              | OpenRLHF           | Mixtral-8x7B-v0.1      | 4.3.0     |
| [Phi-3](models/nlp/llm/phi-3/pytorch)                        | Torchrun           | Phi-3-mini-4k-instruct | 4.2.0     |
| [QWen-7B](models/nlp/llm/qwen-7b/pytorch)                    | Firefly            | qwen-7b                | 3.4.0     |
| [QWen1.5-7B](models/nlp/llm/qwen1.5-7b/pytorch)              | Firefly            | school_math            | 4.1.1     |
| [QWen1.5-14B](models/nlp/llm/qwen1.5-14b/pytorch)            | Firefly            | school_math            | 4.1.1     |
| [Qwen2-7B](models/nlp/llm/qwen2-7b/verl)                     | verl               | qwen2-7b               | dev-only  |
| [Qwen2.5-7B SFT](models/nlp/llm/qwen2.5-7b/pytorch)          | LLaMA-Factory      | qwen2.5-7b             | 4.1.1     |
| [Qwen2.5-1.5B verl](models/nlp/llm/qwen2.5-1.5b/verl)        | verl               | qwen2.5-1.5b           | 4.2.0     |
| [Qwen2.5-7B verl](models/nlp/llm/qwen2.5-7b/verl)            | verl               | qwen2.5-7b             | 4.2.0     |
| [Qwen2.5-3B](models/nlp/llm/qwen2.5-3b/pytorch)              | ColossalAI         | qwen2.5-3b             | 4.3.0     |
| [Qwen2.5-VL-7B](models/nlp/llm/qwen2.5-vl-7b/verl)           | verl               | qwen2.5-vl-7b          | dev-only  |
| [Qwen2.5-VL-7B DPO](models/nlp/llm/qwen2.5-vl-7b/llamafactory) | LLaMA-Factory | Qwen2.5-VL-7B-Instruct | 4.3.0     |
| [Qwen2.5-VL-7B SFT](models/nlp/llm/qwen2.5-vl-7b/llamafactory) | LLaMA-Factory | Qwen2.5-VL-7B-Instruct | 4.3.0     |
| [Qwen3-8B](models/nlp/llm/qwen3-8b/verl)                     | verl               | qwen3-8b               | dev-only  |
| [Yi-6B](models/nlp/llm/yi-6b/pytorch)                        | DeepSpeed          | Yi-6B                  | 4.2.0     |
| [Yi-1.5-6B](models/nlp/llm/yi-1.5-6b/pytorch)                | DeepSpeed          | Yi-1.5-6B              | 4.2.0     |
| [Yi-VL-6B](models/nlp/llm/yi-vl-6b/pytorch)                  | LLaMA-Factory      | Yi-VL-6B-hf            | 4.2.0     |

### Computer Vision

#### Classification

| Model                                                                         | Framework    | Dataset  | IXUCA SDK |
|-------------------------------------------------------------------------------|--------------|----------|-------|
|  [ACmix](models/cv/classification/acmix/pytorch)                                       |  PyTorch       |  ImageNet  | 2.2.0 |
|  [ACNet](models/cv/classification/acnet/pytorch)                                       |  PyTorch       |  ImageNet  | 3.1.0 |
|  [AlexNet](models/cv/classification/alexnet/pytorch)                                   |  PyTorch       |  ImageNet  | 2.2.0 |
|  [AlexNet](models/cv/classification/alexnet/tensorflow)                                |  TensorFlow    |  ImageNet  | 3.1.0 |
|  [BYOL](models/cv/classification/byol/pytorch)                                         |  PyTorch       |  ImageNet  | 3.1.0 |
|  [CBAM](models/cv/classification/cbam/pytorch)                                         |  PyTorch       |  ImageNet  | 3.0.0 |
|  [ConvNext](models/cv/classification/convnext/pytorch)                                 |  PyTorch       |  ImageNet  | 2.2.0 |
|  [CspDarknet53](models/cv/classification/cspdarknet53/pytorch)                         |  PyTorch       |  ImageNet  | 3.0.0 |
|  [DenseNet](models/cv/classification/densenet/paddlepaddle)                            |  PaddlePaddle  |  ImageNet  | 3.1.0 |
|  [DenseNet](models/cv/classification/densenet/pytorch)                                 |  PyTorch       |  ImageNet  | 2.2.0 |
|  [DPN92](models/cv/classification/dpn92/pytorch)                                       |  PyTorch       |  ImageNet  | 2.2.0 |
|  [DPN107](models/cv/classification/dpn107/pytorch)                                     |  PyTorch       |  ImageNet  | 2.2.0 |
|  [ECA-MobileNetV2](models/cv/classification/eca_mobilenet_v2/pytorch)                  |  PyTorch       |  ImageNet  | 2.2.0 |
|  [ECA-ResNet152](models/cv/classification/eca_resnet152/pytorch)                       |  PyTorch       |  ImageNet  | 2.2.0 |
|  [EfficientNetB0](models/cv/classification/efficientnet_b0/paddlepaddle)               |  PaddlePaddle  |  ImageNet  | 3.1.0 |
|  [EfficientNetB4](models/cv/classification/efficientnet_b4/pytorch)                    |  PyTorch       |  ImageNet  | 2.2.0 |
|  [FasterNet](models/cv/classification/fasternet/pytorch)                               |  PyTorch       |  ImageNet  | 3.0.0 |
|  [GoogLeNet](models/cv/classification/googlenet/pytorch)                               |  PyTorch       |  ImageNet  | 2.2.0 |
|  [GoogLeNet](models/cv/classification/googlenet/paddlepaddle)                          |  PaddlePaddle  |  ImageNet  | 2.3.0 |
|  [InceptionV3](models/cv/classification/inceptionv3/mindspore)                         |  MindSpore     |  ImageNet  | 3.1.0 |
|  [InceptionV3](models/cv/classification/inceptionv3/pytorch)                           |  PyTorch       |  ImageNet  | 2.2.0 |
|  [InceptionV3](models/cv/classification/inceptionv3/tensorflow)                        |  TensorFlow    |  ImageNet  | 3.1.0 |
|  [InceptionV4](models/cv/classification/inceptionv4/pytorch)                           |  PyTorch       |  ImageNet  | 2.2.0 |
|  [InternImage](models/cv/classification/internimage/pytorch)                           |  PyTorch       |  ImageNet  | 3.1.0 |
|  [LeNet](models/cv/classification/lenet/pytorch)                                       |  PyTorch       |  ImageNet  | 2.2.0 |
|  [MobileNetV2](models/cv/classification/mobilenetv2/pytorch)                           |  PyTorch       |  ImageNet  | 2.2.0 |
|  [MobileNetV3](models/cv/classification/mobilenetv3/mindspore)                         |  MindSpore     |  ImageNet  | 3.1.0 |
|  [MobileNetV3](models/cv/classification/mobilenetv3/pytorch)                           |  PyTorch       |  ImageNet  | 2.2.0 |
|  [MobileNetV3](models/cv/classification/mobilenetv3/paddlepaddle)                      |  PaddlePaddle  |  ImageNet  | 2.3.0 |
|  [MobileNetV3_Large1.0](models/cv/classification/mobilenetv3_large_x1_0/paddlepaddle)  |  PaddlePaddle  |  ImageNet  | 3.1.0 |
|  [MobileOne](models/cv/classification/mobileone/pytorch)                               |  PyTorch       |  ImageNet  | 3.1.0 |
|  [MoCoV2](models/cv/classification/mocov2/pytorch)                                     |  PyTorch       |  ImageNet  | 3.1.0 |
|  [PP-LCNet](models/cv/classification/pp-lcnet/paddlepaddle)                            |  PaddlePaddle  |  ImageNet  | 3.1.0 |
|  [RepMLP](models/cv/classification/repmlp/pytorch)                                     |  PyTorch       |  ImageNet  | 3.1.0 |
|  [RepVGG](models/cv/classification/repvgg/pytorch)                                     |  PyTorch       |  ImageNet  | 3.0.0 |
|  [RepVGG](models/cv/classification/repvgg/paddlepaddle)                                |  PaddlePaddle  |  ImageNet  | 3.0.0 |
|  [RepViT](models/cv/classification/repvit/pytorch)                                     |  PyTorch       |  ImageNet  | 3.1.0 |
|  [Res2Net50_14w_8s](models/cv/classification/res2net50_14w_8s/paddlepaddle)            |  PaddlePaddle  |  ImageNet  | 3.1.0 |
|  [ResNeSt14](models/cv/classification/resnest14/pytorch)                               |  PyTorch       |  ImageNet  | 2.2.0 |
|  [ResNeSt50](models/cv/classification/resnest50/pytorch)                               |  PyTorch       |  ImageNet  | 2.2.0 |
|  [ResNeSt50](models/cv/classification/resnest50/paddlepaddle)                          |  PaddlePaddle  |  ImageNet  | 3.0.0 |
|  [ResNeSt101](models/cv/classification/resnest101/pytorch)                             |  PyTorch       |  ImageNet  | 2.2.0 |
|  [ResNeSt269](models/cv/classification/resnest269/pytorch)                             |  PyTorch       |  ImageNet  | 2.2.0 |
|  [ResNet18](models/cv/classification/resnet18/pytorch)                                 |  PyTorch       |  ImageNet  | 2.2.0 |
|  [ResNet50](models/cv/classification/resnet50/pytorch)                                 |  PyTorch       |  ImageNet  | 2.2.0 |
|  [ResNet50](models/cv/classification/resnet50/paddlepaddle)                            |  PaddlePaddle  |  ImageNet  | 2.3.0 |
|  [ResNet50](models/cv/classification/resnet50/tensorflow)                              |  TensorFlow    |  ImageNet  | 3.0.0 |
|  [ResNet101](models/cv/classification/resnet101/pytorch)                               |  PyTorch       |  ImageNet  | 2.2.0 |
|  [ResNet152](models/cv/classification/resnet152/pytorch)                               |  PyTorch       |  ImageNet  | 2.2.0 |
|  [ResNeXt50_32x4d](models/cv/classification/resnext50_32x4d/mindspore)                 |  MindSpore     |  ImageNet  | 3.1.0 |
|  [ResNeXt50_32x4d](models/cv/classification/resnext50_32x4d/pytorch)                   |  PyTorch       |  ImageNet  | 2.2.0 |
|  [ResNeXt101_32x8d](models/cv/classification/resnext101_32x8d/pytorch)                 |  PyTorch       |  ImageNet  | 2.2.0 |
|  [SE_ResNet50_vd](models/cv/classification/se_resnet50_vd/paddlepaddle)                |  PaddlePaddle  |  ImageNet  | 3.1.0 |
|  [SEResNeXt](models/cv/classification/seresnext/pytorch)                               |  PyTorch       |  ImageNet  | 2.2.0 |
|  [ShuffleNetV2](models/cv/classification/shufflenetv2/paddlepaddle)                    |  PaddlePaddle  |  ImageNet  | 3.1.0 |
|  [ShuffleNetV2](models/cv/classification/shufflenetv2/pytorch)                         |  PyTorch       |  ImageNet  | 2.2.0 |
|  [SqueezeNet](models/cv/classification/squeezenet/pytorch)                             |  PyTorch       |  ImageNet  | 2.2.0 |
|  [Swin Transformer](models/cv/classification/swin_transformer/paddlepaddle)            |  PaddlePaddle  |  ImageNet  | 3.0.0 |
|  [Swin Transformer](models/cv/classification/swin_transformer/pytorch)                 |  PyTorch       |  ImageNet  | 2.2.0 |
|  [VGG16](models/cv/classification/vgg/paddlepaddle)                                    |  PaddlePaddle  |  ImageNet  | 2.3.0 |
|  [VGG16](models/cv/classification/vgg/pytorch)                                         |  PyTorch       |  ImageNet  | 2.2.0 |
|  [VGG16](models/cv/classification/vgg/tensorflow)                                      |  TensorFlow    |  ImageNet  | 3.1.0 |
|  [Wave-MLP](models/cv/classification/wavemlp/pytorch)                                  |  PyTorch       |  ImageNet  | 2.2.0 |
|  [Wide_ResNet101_2](models/cv/classification/wide_resnet101_2/pytorch)                 |  PyTorch       |  ImageNet  | 2.2.0 |
|  [Xception](models/cv/classification/xception/paddlepaddle)                            |  PaddlePaddle  |  ImageNet  | 3.1.0 |
|  [Xception](models/cv/classification/xception/pytorch)                                 |  PyTorch       |  ImageNet  | 2.2.0 |

#### Face Detection

| Model                                              | Framework | Dataset    | IXUCA SDK |
|----------------------------------------------------|-----------|------------|-------|
|  [RetinaFace](models/cv/face_detection/retinaface/pytorch)  |  PyTorch    |  WIDER FACE  | 3.0.0 |

#### Face Recognition

| Model                                                   | Framework    | Dataset            | IXUCA SDK |
|---------------------------------------------------------|--------------|--------------------|-------|
|  [ArcFace](models/cv/face_recognition/arcface/pytorch)          |  PyTorch       |  CASIA-WebFaces&LFW  | 3.0.0 |
|  [BlazeFace](models/cv/face_recognition/blazeface/paddlepaddle)  |  PaddlePaddle  |  WIDER FACE          | 3.1.0 |
|  [CosFace](models/cv/face_recognition/cosface/pytorch)           |  PyTorch       |  CASIA-WebFaces&LFW  | 3.0.0 |
|  [FaceNet](models/cv/face_recognition/facenet/pytorch)           |  PyTorch       |  CASIA-WebFaces&LFW  | 3.0.0 |
|  [FaceNet](models/cv/face_recognition/facenet/tensorflow)        |  TensorFlow    |  CASIA-WebFaces&LFW  | 3.1.0 |

#### Instance Segmentation

| Model                                               | Framework    | Dataset | IXUCA SDK |
|-----------------------------------------------------|--------------|---------|-------|
|  [SOLO](models/cv/instance_segmentation/solo/pytorch)        |  PyTorch       |  COCO     | 3.0.0 |
|  [SOLOv2](models/cv/detection/solov2/paddlepaddle)           |  PaddlePaddle  |  COCO     | 3.0.0 |
|  [SOLOv2](models/cv/instance_segmentation/solov2/pytorch)    |  PyTorch       |  COCO     | 3.1.0 |
|  [YOLACT++](models/cv/instance_segmentation/yolact/pytorch)  |  PyTorch       |  COCO     | 3.0.0 |

#### Image Generation

| Model                                               | Framework    | Dataset  | IXUCA SDK |
|-----------------------------------------------------|--------------|----------|-------|
|  [DCGAN](models/cv/image_generation/dcgan/mindspore)         |  MindSpore     |  ImageNet  | 3.0.0 |
|  [Pix2Pix](models/cv/image_generation/pix2pix/paddlepaddle)  |  PaddlePaddle  |  facades   | 3.1.0 |

#### Knowledge Distillation

| Model                             | Framework | Dataset      | IXUCA SDK |
|-----------------------------------|-----------|--------------|-------|
|  [CWD](models/cv/distiller/cwd/pytorch)    |  PyTorch    |  Cityscapes    | 3.0.0 |
|  [RKD](models/cv/distiller/rkd/pytorch)    |  PyTorch    |  CUB-200-2011  | 3.0.0 |
|  [WSLD](models/cv/distiller/wsld/pytorch)  |  PyTorch    |  ImageNet      | 3.1.0 |

#### Object Detection

| Model                                                         | Framework             | Dataset | IXUCA SDK |
|---------------------------------------------------------------|-----------------------|---------|-------|
|  [ATSS](models/cv/detection/atss_mmdet/pytorch)                        |  PyTorch (MMDetection)  |  COCO     | 3.0.0 |
|  [AutoAssign](models/cv/detection/autoassign/pytorch)                  |  PyTorch                |  COCO     | 2.2.0 |
|  [Cascade R-CNN](models/cv/detection/cascade_rcnn_mmdet/pytorch)       |  PyTorch (MMDetection)  |  COCO     | 3.0.0 |
|  [CenterMask2](models/cv/detection/centermask2/pytorch)                |  PyTorch                |  COCO     | 4.1.1 |
|  [CenterNet](models/cv/detection/centernet/pytorch)                    |  PyTorch                |  COCO     | 2.2.0 |
|  [CenterNet](models/cv/detection/centernet/paddlepaddle)               |  PaddlePaddle           |  COCO     | 3.0.0 |
|  [Co-DETR](models/cv/detection/co-detr/pytorch)                        |  PyTorch                |  COCO     | 3.1.0 |
|  [CornerNet](models/cv/detection/cornernet_mmdet/pytorch)              |  PyTorch (MMDetection)  |  COCO     | 3.0.0 |
|  [DCNV2](models/cv/detection/dcnv2_mmdet/pytorch)                      |  PyTorch (MMDetection)  |  COCO     | 3.0.0 |
|  [DETR](models/cv/detection/detr/paddlepaddle)                         |  PaddlePaddle           |  COCO     | 3.0.0 |
|  [Faster R-CNN](models/cv/detection/fasterrcnn/pytorch)                |  PyTorch                |  COCO     | 2.2.0 |
|  [FCOS](models/cv/detection/fcos/paddlepaddle)                         |  PaddlePaddle           |  COCO     | 3.0.0 |
|  [FCOS](models/cv/detection/fcos/pytorch)                              |  PyTorch                |  COCO     | 3.0.0 |
|  [Mamba-YOLO](models/cv/detection/mamba_yolo/pytorch)                  |  PyTorch                |  COCO     | 4.1.1 |
|  [Mask R-CNN](models/cv/detection/maskrcnn/pytorch)                    |  PyTorch                |  COCO     | 2.2.0 |
|  [Mask R-CNN](models/cv/detection/maskrcnn/paddlepaddle)               |  PaddlePaddle           |  COCO     | 2.3.0 |
|  [OC_SORT](models/cv/detection/oc_sort/paddlepaddle)                   |  PaddlePaddle           |  MOT17    | 3.1.0 |
|  [Oriented RepPoints](models/cv/detection/oriented_reppoints/pytorch)  |  PyTorch                |  DOTA     | 3.1.0 |
|  [PP-PicoDet](models/cv/detection/picodet/paddlepaddle)                |  PaddlePaddle           |  COCO     | 3.1.0 |
|  [PP-YOLOE](models/cv/detection/pp-yoloe/paddlepaddle)                 |  PaddlePaddle           |  COCO     | 2.3.0 |
|  [PP-YOLOE+](models/cv/detection/pp_yoloe+/paddlepaddle)               |  PaddlePaddle           |  COCO     | 3.1.1 |
|  [PVANet](models/cv/detection/pvanet/pytorch)                          |  PyTorch                |  COCO     | 2.2.0 |
|  [RepPoints](models/cv/detection/reppoints_mmdet/pytorch)              |  PyTorch (MMDetection)  |  COCO     | 3.0.0 |
|  [RetinaNet](models/cv/detection/retinanet/pytorch)                    |  PyTorch                |  COCO     | 2.2.0 |
|  [RetinaNet](models/cv/detection/retinanet/paddlepaddle)               |  PaddlePaddle           |  COCO     | 3.0.0 |
|  [RT-DETR](models/cv/detection/rt-detr/pytorch)                        |  PyTorch                |  COCO     | 4.1.1 |
|  [RTMDet](models/cv/detection/rtmdet/pytorch)                          |  PyTorch                |  COCO     | 3.1.0 |
|  [SSD](models/cv/detection/ssd/pytorch)                                |  PyTorch                |  COCO     | 2.2.0 |
|  [SSD](models/cv/detection/ssd/paddlepaddle)                           |  PaddlePaddle           |  COCO     | 2.3.0 |
|  [SSD](models/cv/detection/ssd/tensorflow)                             |  TensorFlow             |  VOC      | 3.0.0 |
|  [SSD](models/cv/detection/ssd/mindspore)                              |  MindSpore              |  COCO     | 3.0.0 |
|  [YOLOF](models/cv/detection/yolof/pytorch)                            |  PyTorch                |  COCO     | 2.2.0 |
|  [YOLOv3](models/cv/detection/yolov3/pytorch)                          |  PyTorch                |  COCO     | 2.2.0 |
|  [YOLOv3](models/cv/detection/yolov3/paddlepaddle)                     |  PaddlePaddle           |  COCO     | 2.3.0 |
|  [YOLOv3](models/cv/detection/yolov3/tensorflow)                       |  TensorFlow             |  VOC      | 3.0.0 |
|  [YOLOv5](models/cv/detection/yolov5/paddlepaddle)                     |  PaddlePaddle           |  COCO     | 3.1.1 |
|  [YOLOv5](models/cv/detection/yolov5/pytorch)                          |  PyTorch                |  COCO     | 2.2.0 |
|  [YOLOv6](models/cv/detection/yolov6/pytorch)                          |  PyTorch                |  COCO     | 3.0.0 |
|  [YOLOv7](models/cv/detection/yolov7/pytorch)                          |  PyTorch                |  COCO     | 3.0.0 |
|  [YOLOv8](models/cv/detection/yolov8/pytorch)                          |  PyTorch                |  COCO     | 3.0.0 |
|  [YOLOv9](models/cv/detection/yolov9/pytorch)                          |  PyTorch                |  COCO     | 4.1.1 |
|  [YOLOv10](models/cv/detection/yolov10/pytorch)                        |  PyTorch                |  COCO     | 4.1.1 |
|  [YOLOv11](models/cv/detection/yolov11/pytorch)                        |  PyTorch                |  COCO     | 4.2.0 |

#### 3D Object Detection

| Model                                                         | Framework | Dataset          | IXUCA SDK |
|---------------------------------------------------------------|-----------|------------------|-------|
|  [BEVFormer](models/cv/3d_detection/bevformer/pytorch)                 |  PyTorch    |  nuScenes&CAN bus  | 3.0.0 |
|  [CenterPoint](models/cv/3d_detection/centerpoint/pytorch)             |  PyTorch    |  nuScenes          | 3.1.1 |
|  [PAConv](models/cv/3d_detection/paconv/pytorch)                       |  PyTorch    |  S3DIS             | 3.1.1 |
|  [Part-A2-Anchor](models/cv/3d_detection/part_a2_anchor/pytorch)       |  PyTorch    |  KITTI             | 4.1.1 |
|  [Part-A2-Free](models/cv/3d_detection/part_a2_free/pytorch)           |  PyTorch    |  KITTI             | 4.1.1 |
|  [PointNet++](models/cv/3d_detection/pointnet2/pytorch)                |  PyTorch    |  S3DIS             | 3.0.0 |
|  [PointPillars](models/cv/3d_detection/pointpillars/pytorch)           |  PyTorch    |  KITTI             | 3.0.0 |
|  [PointRCNN](models/cv/3d_detection/pointrcnn/pytorch)                 |  PyTorch    |  KITTI             | 3.1.1 |
|  [PointRCNN-IoU](models/cv/3d_detection/pointrcnn_iou/pytorch)         |  PyTorch    |  KITTI             | 4.1.1 |
|  [SECOND](models/cv/3d_detection/second/pytorch)                       |  PyTorch    |  KITTI             | 4.1.1 |
|  [SECOND-IoU](models/cv/3d_detection/second_iou/pytorch)               |  PyTorch    |  KITTI             | 4.1.1 |

#### 3D Reconstruction

| Model                                             | Framework | Dataset | IXUCA SDK |
|---------------------------------------------------|-----------|---------|-------|
|  [HashNeRF](models/cv/3d-reconstruction/hashnerf/pytorch)  |  PyTorch    |  fox      | 2.2.0 |

#### GNN (Graph Neural Network)

| Model                                      | Framework    | Dataset                  | IXUCA SDK |
|--------------------------------------------|--------------|--------------------------|-------|
|  [GAT](models/cv/gnn/gat/paddlepaddle)              |  PaddlePaddle  |  CORA                      | 3.1.0 |
|  [GCN](models/cv/gnn/GCN/mindspore)                 |  MindSpore     |  CORA & Citeseer           | 3.0.0 |
|  [GCN](models/cv/gnn/GCN/paddlepaddle)              |  PaddlePaddle  |  CORA & PubMed & Citeseer  | 3.1.0 |
|  [GraphSAGE](models/cv/gnn/graphsage/paddlepaddle)  |  PaddlePaddle  |  Reddit                    | 3.1.0 |

#### OCR

| Model                                          | Framework    | Dataset   | IXUCA SDK |
|------------------------------------------------|--------------|-----------|-------|
|  [CRNN](models/cv/ocr/crnn/mindspore)                   |  MindSpore     |  OCR_Recog  | 3.1.0 |
|  [CRNN](models/cv/ocr/crnn/paddlepaddle)                |  PaddlePaddle  |  LMDB       | 2.3.0 |
|  [DBNet](models/cv/ocr/dbnet/pytorch)                   |  PyTorch       |  ICDAR2015  | 3.0.0 |
|  [DBNet++](models/cv/ocr/dbnetpp/paddlepaddle)          |  PaddlePaddle  |  ICDAR2015  | 3.1.1 |
|  [DBNet++](models/cv/ocr/dbnetpp/pytorch)               |  PyTorch       |  ICDAR2015  | 3.1.0 |
|  [PP-OCR-DB](models/cv/ocr/pp-ocr-db/paddlepaddle)      |  PaddlePaddle  |  ICDAR2015  | 2.3.0 |
|  [PP-OCR-EAST](models/cv/ocr/pp-ocr-east/paddlepaddle)  |  PaddlePaddle  |  ICDAR2015  | 3.1.1 |
|  [PSE](models/cv/ocr/pse/paddlepaddle)                  |  PaddlePaddle  |  OCR_Recog  | 2.3.0 |
|  [SAR](models/cv/ocr/sar/pytorch)                       |  PyTorch       |  OCR_Recog  | 2.2.0 |
|  [SAST](models/cv/ocr/sast/paddlepaddle)                |  PaddlePaddle  |  ICDAR2015  | 3.1.1 |
|  [SATRN](models/cv/ocr/satrn/pytorch/base)              |  PyTorch       |  OCR_Recog  | 2.2.0 |

#### Point Cloud

| Model                                           | Framework | Dataset                         | IXUCA SDK |
|-------------------------------------------------|-----------|---------------------------------|-------|
|  [Point-BERT](models/cv/point_cloud/point-bert/pytorch)  |  PyTorch    |  ShapeNet55 & processed ModelNet  | 2.2.0 |

#### Pose Estimation

| Model                                   | Framework    | Dataset | IXUCA SDK |
|-----------------------------------------|--------------|---------|-------|
|  [AlphaPose](models/cv/pose/alphapose/pytorch)   |  PyTorch       |  COCO     | 3.0.0 |
|  [HRNet](models/cv/pose/hrnet/pytorch)           |  PyTorch       |  COCO     | 2.2.0 |
|  [HRNet-W32](models/cv/pose/hrnet/paddlepaddle)  |  PaddlePaddle  |  COCO     | 3.1.0 |
|  [OpenPose](models/cv/pose/openpose/mindspore)   |  MindSpore     |  COCO     | 3.1.0 |

#### Self-Supervised Learning

| Model                                          | Framework | Dataset  | IXUCA SDK |
|------------------------------------------------|-----------|----------|-------|
|  [MAE](models/cv/self_supervised_learning/mae/pytorch)  |  PyTorch    |  ImageNet  | 3.0.0 |

#### Semantic Segmentation

| Model                                                                | Framework    | Dataset        | IXUCA SDK |
|----------------------------------------------------------------------|--------------|----------------|-------|
|  [3D-UNet](models/cv/semantic_segmentation/unet3d/pytorch)                    |  PyTorch       |  kits19          | 2.2.0 |
|  [APCNet](models/cv/semantic_segmentation/apcnet/pytorch)                     |  PyTorch       |  Cityscapes      | 2.2.0 |
|  [Attention U-net](models/cv/semantic_segmentation/att_unet/pytorch)          |  PyTorch       |  Cityscapes      | 3.0.0 |
|  [BiSeNet](models/cv/semantic_segmentation/bisenet/pytorch)                   |  PyTorch       |  COCO            | 2.2.0 |
|  [BiSeNetV2](models/cv/semantic_segmentation/bisenetv2/paddlepaddle)          |  PaddlePaddle  |  Cityscapes      | 3.0.0 |
|  [BiSeNetV2](models/cv/semantic_segmentation/bisenetv2/pytorch)               |  PyTorch       |  Cityscapes      | 3.1.1 |
|  [CGNet](models/cv/semantic_segmentation/cgnet/pytorch)                       |  PyTorch       |  COCO            | 2.2.0 |
|  [ContextNet](models/cv/semantic_segmentation/contextnet/pytorch)             |  PyTorch       |  COCO            | 2.2.0 |
|  [DabNet](models/cv/semantic_segmentation/dabnet/pytorch)                     |  PyTorch       |  COCO            | 2.2.0 |
|  [DANet](models/cv/semantic_segmentation/danet/pytorch)                       |  PyTorch       |  COCO            | 2.2.0 |
|  [DDRnet](models/cv/semantic_segmentation/ddrnet/pytorch)                     |  PyTorch       |  Cityscapes      | 3.0.0 |
|  [DeepLabV3](models/cv/semantic_segmentation/deeplabv3/pytorch)               |  PyTorch       |  COCO            | 2.2.0 |
|  [DeepLabV3](models/cv/semantic_segmentation/deeplabv3/paddlepaddle)          |  PaddlePaddle  |  Cityscapes      | 2.3.0 |
|  [DeepLabV3](models/cv/semantic_segmentation/deeplabv3/mindspore)             |  MindSpore     |  VOC             | 3.0.0 |
|  [DeepLabV3+](models/cv/semantic_segmentation/deeplabv3plus/paddlepaddle)     |  PaddlePaddle  |  Cityscapes      | 3.0.0 |
|  [DeepLabV3+](models/cv/semantic_segmentation/deeplabv3plus/tensorflow)       |  TensorFlow    |  Cityscapes      | 3.1.0 |
|  [DenseASPP](models/cv/semantic_segmentation/denseaspp/pytorch)               |  PyTorch       |  COCO            | 2.2.0 |
|  [DFANet](models/cv/semantic_segmentation/dfanet/pytorch)                     |  PyTorch       |  COCO            | 2.2.0 |
|  [DNLNet](models/cv/semantic_segmentation/dnlnet/paddlepaddle)                |  PaddlePaddle  |  Cityscapes      | 2.3.0 |
|  [DUNet](models/cv/semantic_segmentation/dunet/pytorch)                       |  PyTorch       |  COCO            | 2.2.0 |
|  [EncNet](models/cv/semantic_segmentation/encnet/pytorch)                     |  PyTorch       |  COCO            | 2.2.0 |
|  [ENet](models/cv/semantic_segmentation/enet/pytorch)                         |  PyTorch       |  COCO            | 2.2.0 |
|  [ERFNet](models/cv/semantic_segmentation/erfnet/pytorch)                     |  PyTorch       |  COCO            | 2.2.0 |
|  [ESPNet](models/cv/semantic_segmentation/espnet/pytorch)                     |  PyTorch       |  COCO            | 2.2.0 |
|  [FastFCN](models/cv/semantic_segmentation/fastfcn/paddlepaddle)              |  PyTorch       |  ADE20K          | 3.1.0 |
|  [FastSCNN](models/cv/semantic_segmentation/fastscnn/pytorch)                 |  PyTorch       |  COCO            | 2.2.0 |
|  [FCN](models/cv/semantic_segmentation/fcn/pytorch)                           |  PyTorch       |  COCO            | 2.2.0 |
|  [FPENet](models/cv/semantic_segmentation/fpenet/pytorch)                     |  PyTorch       |  COCO            | 2.2.0 |
|  [GCNet](models/cv/semantic_segmentation/gcnet/pytorch)                       |  PyTorch       |  Cityscapes      | 2.2.0 |
|  [HardNet](models/cv/semantic_segmentation/hardnet/pytorch)                   |  PyTorch       |  COCO            | 2.2.0 |
|  [ICNet](models/cv/semantic_segmentation/icnet/pytorch)                       |  PyTorch       |  COCO            | 2.2.0 |
|  [LedNet](models/cv/semantic_segmentation/lednet/pytorch)                     |  PyTorch       |  COCO            | 2.2.0 |
|  [LinkNet](models/cv/semantic_segmentation/linknet/pytorch)                   |  PyTorch       |  COCO            | 2.2.0 |
|  [Mask2Former](models/cv/semantic_segmentation/mask2former/pytorch)           |  PyTorch       |  Cityscapes      | 3.1.0 |
|  [MobileSeg](models/cv/semantic_segmentation/mobileseg/paddlepaddle)          |  PaddlePaddle  |  Cityscapes      | 3.1.0 |
|  [OCNet](models/cv/semantic_segmentation/ocnet/pytorch)                       |  PyTorch       |  COCO            | 2.2.0 |
|  [OCRNet](models/cv/semantic_segmentation/ocrnet/paddlepaddle)                |  PaddlePaddle  |  Cityscapes      | 3.1.0 |
|  [OCRNet](models/cv/semantic_segmentation/ocrnet/pytorch)                     |  PyTorch       |  Cityscapes      | 2.2.0 |
|  [PP-HumanSegV1](models/cv/semantic_segmentation/pp_humansegv1/paddlepaddle)  |  PaddlePaddle  |  PP-HumanSeg14K  | 3.1.0 |
|  [PP-HumanSegV2](models/cv/semantic_segmentation/pp_humansegv2/paddlepaddle)  |  PaddlePaddle  |  PP-HumanSeg14K  | 3.1.0 |
|  [PP-LiteSeg](models/cv/semantic_segmentation/pp_liteseg/paddlepaddle)        |  PaddlePaddle  |  Cityscapes      | 3.1.0 |
|  [PSANet](models/cv/semantic_segmentation/psanet/pytorch)                     |  PyTorch       |  COCO            | 2.2.0 |
|  [PSPNet](models/cv/semantic_segmentation/pspnet/pytorch)                     |  PyTorch       |  Cityscapes      | 2.2.0 |
|  [RefineNet](models/cv/semantic_segmentation/refinenet/pytorch)               |  PyTorch       |  COCO            | 2.2.0 |
|  [SegNet](models/cv/semantic_segmentation/segnet/pytorch)                     |  PyTorch       |  COCO            | 2.2.0 |
|  [STDC](models/cv/semantic_segmentation/stdc/paddlepaddle)                    |  PaddlePaddle  |  Cityscapes      | 3.1.0 |
|  [STDC](models/cv/semantic_segmentation/stdc/pytorch)                         |  PyTorch       |  Cityscapes      | 3.0.0 |
|  [UNet](models/cv/semantic_segmentation/unet/pytorch)                         |  PyTorch       |  COCO            | 2.2.0 |
|  [UNet](models/cv/semantic_segmentation/unet/paddlepaddle)                    |  PaddlePaddle  |  Cityscapes      | 2.3.0 |
|  [UNet++](models/cv/semantic_segmentation/unet++/pytorch)                     |  PyTorch       |  DRIVE           | 3.0.0 |
|  [VNet](models/cv/semantic_segmentation/vnet/tensorflow)                      |  TensorFlow    |  Hippocampus     | 3.0.0 |

#### Super Resolution

| Model                                                     | Framework | Dataset | IXUCA SDK |
|-----------------------------------------------------------|-----------|---------|-------|
|  [basicVSR++](models/cv/super_resolution/basicvsr++/pytorch)       |  PyTorch    |  REDS     | 2.2.0 |
|  [basicVSR](models/cv/super_resolution/basicvsr/pytorch)           |  PyTorch    |  REDS     | 2.2.0 |
|  [ESRGAN](models/cv/super_resolution/esrgan/pytorch)               |  PyTorch    |  DIV2K    | 2.2.0 |
|  [LIIF](models/cv/super_resolution/liif/pytorch)                   |  PyTorch    |  DIV2K    | 2.2.0 |
|  [RealBasicVSR](models/cv/super_resolution/real_basicvsr/pytorch)  |  PyTorch    |  REDS     | 2.2.0 |
|  [TTSR](models/cv/super_resolution/ttsr/pytorch)                   |  PyTorch    |  CUFED    | 2.2.0 |
|  [TTVSR](models/cv/super_resolution/ttvsr/pytorch)                 |  PyTorch    |  REDS     | 2.2.0 |

#### Multi-Object Tracking

| Model                                           | Framework    | Dataset     | IXUCA SDK |
|-------------------------------------------------|--------------|-------------|-------|
|  [ByteTrack](models/cv/multi_object_tracking/bytetrack/paddlepaddle)  |  PaddlePaddle  |  MOT17        | 3.1.0 |
|  [DeepSORT](models/cv/multi_object_tracking/deep_sort/pytorch)        |  PyTorch       |  Market-1501  | 3.0.0 |
|  [FairMOT](models/cv/multi_object_tracking/fairmot/pytorch)           |  PyTorch       |  MOT17        | 2.2.0 |

### Multimodal

| Model                                                                                       | Framework | Dataset        | IXUCA SDK |
|---------------------------------------------------------------------------------------------|-----------|----------------|-----------|
| [BLIP](models/multimodal/vision-language_model/blip/pytorch)                                       | PyTorch   | COCO           | 3.1.1     |
| [CLIP](models/multimodal/contrastive_learning/clip/pytorch)                                        | PyTorch   | CIFAR100       | 2.2.0     |
| [ControlNet](models/multimodal/diffusion_model/controlnet/pytorch)                                 | PyTorch   | Fill50K        | 3.1.0     |
| [DDPM](models/multimodal/diffusion_model/ddpm/pytorch)                                             | PyTorch   | CIFAR-10       | 3.1.0     |
| [LLaVA 1.5](models/multimodal/vision-language_model/llava-1.5/pytorch)                             | PyTorch   | LLaVA-Pretrain | 4.1.1     |
| [L-Verse](models/multimodal/vision-language_model/l-verse/pytorch)                                 | PyTorch   | ImageNet       | 2.2.0     |
| [MoE-LLaVA-Phi2-2.7B](models/multimodal/vision-language_model/moe-llava-phi2-2.7b/pytorch)         | PyTorch   | MoE-LLaVA      | 4.2.0     |
| [MoE-LLaVA-Qwen-1.8B](models/multimodal/vision-language_model/moe-llava-qwen-1.8b/pytorch)         | PyTorch   | MoE-LLaVA      | 4.2.0     |
| [MoE-LLaVA-StableLM-1.6B](models/multimodal/vision-language_model/moe-llava-stablelm-1.6b/pytorch) | PyTorch   | MoE-LLaVA      | 4.2.0     |
| [Stable Diffusion 1.4](models/multimodal/diffusion_model/stable-diffusion-1.4/pytorch)             | PyTorch   | pokemon-images | 3.0.0     |
| [Stable Diffusion 1.5](models/multimodal/diffusion_model/stable-diffusion-1.5/pytorch)             | PyTorch   | pokemon-images | 4.1.1     |
| [Stable Diffusion 2.1](models/multimodal/diffusion_model/stable-diffusion-2.1/pytorch)             | PyTorch   | pokemon-images | 4.1.1     |
| [Stable Diffusion 3](models/multimodal/diffusion_model/stable-diffusion-3/pytorch)                 | PyTorch   | dog-example    | 4.1.1     |
| [Stable Diffusion XL](models/multimodal/diffusion_model/stable-diffusion-xl/pytorch)               | PyTorch   | pokemon-images | 4.1.1     |

### NLP (Natural Language Processing)

#### Cloze Test

| Model                                                   | Framework | Dataset               | IXUCA SDK |
|---------------------------------------------------------|-----------|-----------------------|-------|
|  [GLM](models/nlp/cloze_test/glm/pytorch)  |  PyTorch    |  GLMForMultiTokenCloze  | 2.2.0 |

#### Dialogue Generation

| Model                                      | Framework | Dataset | IXUCA SDK |
|--------------------------------------------|-----------|---------|-------|
|  [CPM](models/nlp/dialogue_generation/cpm/pytorch)  |  PyTorch    |  STC      | 2.2.0 |

#### Language Modeling

| Model                                                            | Framework         | Dataset            | IXUCA SDK |
|------------------------------------------------------------------|-------------------|--------------------|-------|
|  [BART](models/nlp/language_model/bart_fairseq/pytorch)                   |  PyTorch (Fairseq)  |  RTE                 | 3.0.0 |
|  [BERT NER](models/nlp/ner/bert/pytorch)                                  |  PyTorch            |  CoNLL-2003          | 3.0.0 |
|  [BERT Pretraining](models/nlp/language_model/bert_sample/pytorch)        |  PyTorch            |  bert_mini           | 4.3.0 |
|  [BERT Pretraining](models/nlp/language_model/bert/pytorch)               |  PyTorch            |  MLCommon Wikipedia  | 2.2.0 |
|  [BERT Pretraining](models/nlp/language_model/bert/paddlepaddle)          |  PaddlePaddle       |  MNLI                | 2.3.0 |
|  [BERT Pretraining](models/nlp/language_model/bert/tensorflow)            |  TensorFlow         |  MNLI                | 3.0.0 |
|  [BERT Pretraining](models/nlp/language_model/bert/mindspore)             |  MindSpore          |  SQuAD               | 3.0.0 |
|  [BERT Text Classification](models/nlp/text_classification/bert/pytorch)  |  PyTorch            |  GLUE                | 3.0.0 |
|  [BERT Text Summerization](models/nlp/text_summarisation/bert/pytorch)    |  PyTorch            |  cnn_dailymail       | 3.0.0 |
|  [BERT Question Answering](models/nlp/question_answering/bert/pytorch)    |  PyTorch            |  SQuAD               | 3.0.0 |
|  [GPT2-Medium-EN](models/nlp/llm/gpt2-medium-en/paddlepaddle)             |  PaddlePaddle       |  SST-2               | 3.1.0 |
|  [RoBERTa](models/nlp/language_model/roberta_fairseq/pytorch)             |  PyTorch (Fairseq)  |  RTE                 | 3.0.0 |
|  [XLNet](models/nlp/language_model/xlnet/paddlepaddle)                    |  PaddlePaddle       |  SST-2               | 3.1.0 |

#### Text Correction

| Model                                           | Framework    | Dataset | IXUCA SDK |
|-------------------------------------------------|--------------|---------|-------|
|  [ERNIE](models/nlp/text_correction/ernie/paddlepaddle)  |  PaddlePaddle  |  corpus   | 2.3.0 |

#### Translation

| Model                                                          | Framework         | Dataset | IXUCA SDK |
|----------------------------------------------------------------|-------------------|---------|-------|
|  [Convolutional](models/nlp/translation/convolutional_fairseq/pytorch)  |  PyTorch (Fairseq)  |  WMT14    | 3.0.0 |
|  [T5](models/nlp/translation/t5/pytorch)                                |  PyTorch            |  WMT14    | 2.2.0 |
|  [Transformer](models/nlp/translation/transformer/paddlepaddle)         |  PaddlePaddle       |  WMT14    | 2.3.0 |
|  [Transformer](models/nlp/translation/transformer_fairseq/pytorch)      |  PyTorch (Fairseq)  |  IWSLT14  | 3.0.0 |

### Reinforcement Learning

| Model                                                              | Framework    | Dataset     | IXUCA SDK |
|--------------------------------------------------------------------|--------------|-------------|-------|
|  [DQN](models/reinforcement_learning/q-learning-networks/dqn/paddlepaddle)  |  PaddlePaddle  |  CartPole-v0  | 3.1.0 |

### Audio

#### Speech Recognition

| Model                                                                                   | Framework       | Dataset  | IXUCA SDK |
|-----------------------------------------------------------------------------------------|-----------------|----------|-------|
|  [Conformer](models/audio/speech_recognition/conformer_wenet/pytorch)                            |  PyTorch (WeNet)  |  AISHELL   | 2.2.0 |
|  [Conformer](models/audio/speech_recognition/conformer/pytorch)                                  |  PyTorch          | LibriSpeech | 4.3.0 |
|  [Efficient Conformer v2](models/audio/speech_recognition/efficient_conformer_v2_wenet/pytorch)  |  PyTorch (WeNet)  |  AISHELL   | 3.1.0 |
|  [PP-ASR-Conformer](models/audio/speech_recognition/conformer/paddlepaddle)                      |  PaddlePaddle     |  AISHELL   | 3.1.0 |
|  [RNN-T](models/audio/speech_recognition/rnnt/pytorch)                                           |  PyTorch          |  LJSpeech  | 2.2.0 |
|  [Transformer](models/audio/speech_recognition/transformer_wenet/pytorch)                        |  PyTorch (WeNet)  |  AISHELL   | 3.0.0 |
|  [U2++ Conformer](models/audio/speech_recognition/u2++_conformer_wenet/pytorch)                  |  PyTorch (WeNet)  |  AISHELL   | 3.0.0 |
|  [Unified Conformer](models/audio/speech_recognition/unified_conformer_wenet/pytorch)            |  PyTorch (WeNet)  |  AISHELL   | 3.0.0 |

#### Speech Synthesis

| Model                                                                 | Framework    | Dataset     | IXUCA SDK |
|-----------------------------------------------------------------------|--------------|-------------|-------|
|  [CosyVoice2-0.5B](models/audio/speech_synthesis/cosyvoice/pytorch)            |  DeepSpeed     |  openslr      | 4.3.0 |
|  [PP-TTS-FastSpeech2](models/audio/speech_synthesis/fastspeech2/paddlepaddle)  |  PaddlePaddle  |  CSMSC        | 3.1.0 |
|  [PP-TTS-HiFiGAN](models/audio/speech_synthesis/hifigan/paddlepaddle)          |  PaddlePaddle  |  CSMSC        | 3.1.0 |
|  [Tacotron2](models/audio/speech_synthesis/tacotron2/pytorch)                  |  PyTorch       |  LJSpeech     | 2.2.0 |
|  [VQMIVC](models/audio/speech_synthesis/vqmivc/pytorch)                        |  PyTorch       |  VCTK-Corpus  | 2.2.0 |
|  [WaveGlow](models/audio/speech_synthesis/waveglow/pytorch)                    |  PyTorch       |  LJSpeech     | 2.2.0 |

### Others

#### Graph Machine Learning

| Model                                                                | Framework | Dataset            | IXUCA SDK |
|----------------------------------------------------------------------|-----------|--------------------|-------|
|  [Graph WaveNet](models/others/graph_machine_learning/graph_wavenet/pytorch)  |  PyTorch    |  METR-LA & PEMS-BAY  | 2.2.0 |

#### Kolmogorov-Arnold Networks

| Model                                                | Framework | Dataset | IXUCA SDK |
|------------------------------------------------------|-----------|---------|-------|
|  [KAN](models/others/kolmogorov_arnold_networks/kan/pytorch)  |  PyTorch    |  -        | 4.1.1 |

#### Model Pruning

| Model                                                             | Framework | Dataset      | IXUCA SDK |
|-------------------------------------------------------------------|-----------|--------------|-------|
|  [Network Slimming](models/others/model_pruning/network-slimming/pytorch)  |  PyTorch    |  CIFAR-10/100  | 3.0.0 |

#### Recommendation Systems

| Model                                                             | Framework    | Dataset         | IXUCA SDK |
|-------------------------------------------------------------------|--------------|-----------------|-------|
|  [DeepFM](models/others/recommendation_systems/deepfm/paddlepaddle)        |  PaddlePaddle  |  Criteo_Terabyte  | 2.3.0 |
|  [DLRM](models/others/recommendation_systems/dlrm/pytorch)                 |  PyTorch       |  Criteo_Terabyte  | 2.2.0 |
|  [DLRM](models/others/recommendation_systems/dlrm/paddlepaddle)            |  PaddlePaddle  |  Criteo_Terabyte  | 3.1.0 |
|  [FFM](models/others/recommendation_systems/ffm/paddlepaddle)              |  PaddlePaddle  |  Criteo_Terabyte  | 3.1.0 |
|  [NCF](models/others/recommendation_systems/ncf/pytorch)                   |  PyTorch       |  movielens        | 2.2.0 |
|  [Wide&Deep](models/others/recommendation_systems/wide_deep/paddlepaddle)  |  PaddlePaddle  |  Criteo_Terabyte  | 2.3.0 |
|  [xDeepFM](models/others/recommendation_systems/xdeepfm/paddlepaddle)      |  PaddlePaddle  |  Criteo_Terabyte  | 3.1.0 |

--------

## Community

### Code of Conduct

Please refer to DeepSpark Code of Conduct on
[Gitee](https://gitee.com/deep-spark/deepspark/blob/master/CODE_OF_CONDUCT.md) or on
[GitHub](https://github.com/Deep-Spark/deepspark/blob/main/CODE_OF_CONDUCT.md).

### Contact

Please contact <contact@deepspark.org.cn>.

### Contribution

Please refer to the [DeepSparkHub Contributing Guidelines](CONTRIBUTING.md).

### Disclaimers

DeepSparkHub only provides download and preprocessing scripts for public datasets. These datasets do not belong to
DeepSparkHub, and DeepSparkHub is not responsible for their quality or maintenance. Please ensure that you have the
necessary usage licenses for these datasets. Models trained based on these datasets can only be used for non-commercial
research and education purposes.

To dataset owners:

If you do not want your dataset to be published on DeepSparkHub or wish to update the dataset that belongs to you on
DeepSparkHub, please submit an issue on Gitee or Github. We will delete or update it according to your issue. We
sincerely appreciate your support and contributions to our community.

## License

This project is released under [Apache-2.0](LICENSE) License.
