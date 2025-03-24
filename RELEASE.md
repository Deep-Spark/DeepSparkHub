# DeepSparkHub Release Notes

## DeepSparkHub 25.03 Release Notes

### 模型与算法

* 新增了9个大模型训练示例，涉及DeepSpeed，MoE-LLaVA和LLaMA-Factory工具箱

<table>
    </tr>
        <tr align="left"><th colspan=5>大模型</th></tr>
    <tr>
        <td>GLM-4</td>
        <td>MiniCPM(DeepSpeed)</td>
        <td>Phi-3</td>
    </tr>
    <tr>
        <td>MoE-LLaVA-Phi2-2.7B</td>
        <td>MoE-LLaVA-Qwen-1.8B</td>
        <td>MoE-LLaVA-StableLM-1.6B</td>
    </tr>
    <tr>
        <td>Yi-6B (DeepSpeed)</td>
        <td>Yi-1.5-6B (DeepSpeed)</td>
        <td>Yi-VL-6B (LLaMA-Factory)</td>
    </tr>
</table>

### 问题修复

* 同步了Tacotron2 PyTorch模型的最新代码。
* 删除了ConvNeXt，Co-DETR和CenterNet等模型的冗余代码，并对齐社区版本。
* 更新了MMDetection工具箱版本至v3.3.0，并同步ATSS、Cascade R-CNN、CornerNet等模型代码。
* 增加了cv/classification和cv/detection的自动化CI脚本。
* 更新了所有模型README文档格式，补充了模型所支持的IXUCA SDK版本。

### 版本关联

DeepSparkHub 25.03对应天数软件栈4.2.0版本。

### 贡献者

感谢以下社区贡献者

李一力，sanghui-ilu，majorli6，honglyua，吴永乐。

欢迎以任何形式为DeepSparkHub项目贡献。

## DeepSparkHub 24.12 Release Notes

### 特性和增强

#### 模型与算法

* 新增了4个PyTorch训练模型，其中2个为Multimodal模型
* 新增了5个大模型训练示例，涉及ColossalAI，Megatron-LM和LLaMA-Factory

<table>
    <tr align="left"><th colspan=5>PyTorch</th></tr>
    <tr>
        <td>KAN</td>
        <td>Mamba-YOLO</td>
        <td>LLaVA 1.5</td>
    </tr>
    <tr>
        <td>Stable Diffusion 3</td>
        <td></td>
        <td></td>
    </tr>
        <tr align="left"><th colspan=5>大模型</th></tr>
    <tr>
        <td>DeepSeek MoE 7B (ColossalAI)</td>
        <td>Llama3-8B SFT (ColossalAI)</td>
        <td>Mamba-2 (Megatron-LM)</td>
    </tr>
    <tr>
        <td>Mixtral 8x7B (Megatron-LM)</td>
        <td>Qwen2.5-7B SFT (LLaMA-Factory)</td>
        <td></td>
    </tr>
</table>

### 问题修复

* 更新了Llama2-7B (Megatron-DeepSpeed)模型的数据集链接。
* 同步了DBNet、Fast R-CNN和GoogLeNet等PyTorch模型的代码。
* 同步了BERT Base、FaceNet和ResNet50等TensorFlow模型的代码。
* 同步了ClossalAI、Megatron-DeepSpeed和OpenPCDet工具箱的代码。

### 版本关联

DeepSparkHub 24.12对应天数软件栈4.1.1版本。

### 贡献者

感谢以下人员做出的贡献：

李一力，sanghui-ilu，majorli6，honglyua，吴永乐。

欢迎以任何形式为DeepSparkHub项目贡献。

---

## DeepSparkHub 24.09 Release Notes

### 特性和增强

#### 模型与算法

* 新增了5个PyTorch训练模型，其中3个为Stable Diffusion模型

<table>
    <tr align="left"><th colspan=5>PyTorch</th></tr>
    <tr>
        <td>Stable Diffusion 1.5</td>
        <td>Stable Diffusion 2.1</td>
        <td>Stable Diffusion XL</td>
    </tr>
    <tr>
        <td>CenterMask2</td>
        <td>YOLOv10</td>
        <td></td>
    </tr>
</table>

* 新增了4个大语言模型相关训练示例。
  * ChatGLM3-6B (DeepSpeed)
  * Llama3-8B (Megatron-DeepSpeed)
  * Qwen1.5-7B (Firefly)
  * Qwen1.5-14B (Firefly)

### 问题修复

* 修复了HardNet PyTorch模型的torchvision导入报错的问题。
* 同步了AlexNet，EfficientNet_B4等PyTorch模型的代码。
* 同步了AlexNet，FaceNet，InceptionV3，VGG16等TensorFlow模型的代码。
* 同步了工具箱Megatron-Deepspeed大语言模型框架的代码。

### 版本关联

DeepSparkHub 24.09对应天数软件栈4.1.1版本。

### 贡献者

感谢以下人员做出的贡献：

李一力，sanghui-ilu，majorli，Jino Yang，吴永乐。

欢迎以任何形式为DeepSparkHub项目贡献。

---

## DeepSparkHub 24.06 Release Notes

### 特性和增强

#### 模型与算法

* 新增了7个PyTorch训练模型和对[OpenPCDet](https://github.com/open-mmlab/OpenPCDet)工具箱的支持

<table>
    <tr align="left"><th colspan=5>PyTorch</th></tr>
    <tr>
        <td>Part-A2-Anchor</td>
        <td>Part-A2-Free</td>
        <td>PointRCNN-IoU</td>
        <td>RT-DETR</td>
    </tr>
    <tr>
        <td>SECOND</td>
        <td>SECOND-IoU</td>
        <td>YOLOv9</td>
        <td></td>
    </tr>
</table>

* 新增了8个大语言模型相关训练示例。
  * Aquila2-34B (Megatron-DeepSpeed)
  * Baichuan2-7B SFT (DeepSpeed)
  * Bloom-7B1 SFT LoRA (Firefly)
  * ChatGLM2-6B SFT (DeepSpeed)
  * Llama2-7B RLHF (Megatron-DeepSpeed)
  * Llama2-13B (Megatron-DeepSpeed)
  * Llama2-34B (Megatron-DeepSpeed)
  * Qwen-7B SFT LoRA Ptuning-V2 (Firefly)

### 问题修复

* 修复了CenterNet PyTorch模型的DCNv2 版本不配套的问题。
* 修复了BERT Pytorch模型的h5py依赖库版本不对的问题。
* 修复了YOLOv7模型在PyTorch 2.x版本训练启动报错的问题。
* 同步了YOLOv5，FaceNet，AlexNet，InceptionV3，VGG16模型的代码。
* 完善了BiSeNetV2，DeepLabV3，SSD，YOLOv5模型的说明文档。

### 版本关联

DeepSparkHub 24.06对应天数软件栈3.4.0版本。

### 贡献者

感谢以下人员做出的贡献：

majorli，Jino Yang，李一力，shengquan.nian，wenfeng.zhang，吴永乐，zhaojun0044，xiaomei.wang，gaiqin.bai，sanghui-ilu。

欢迎以任何形式为DeepSparkHub项目贡献。

---

## DeepSparkHub 24.03 Release Notes

### 特性和增强

#### 模型与算法

* 新增了10个算法模型。

<table>
    <tr align="left"><th colspan=6>PyTorch</th></tr>
    <tr>
        <td>BiSeNetV2</td>
        <td>BLIP</td>
        <td>CenterPoint</td>
        <td>PAConv</td>
        <td>PointRCNN</td>
    </tr>
    <tr align="left"><th colspan=6>PaddlePaddle</th></tr>
    <tr>
        <td>DBNet++</td>
        <td>PP-OCR-EAST</td>
        <td>PP-YOLOE+</td>
        <td>SAST</td>
        <td>YOLOv5</td>    
    </tr>
</table>

* 新增了基于分布式训练框架的大语言模型训练示例。
    * Megatron-DeepSpeed Llama-2-7B SFT
    * DeepSpeed Llama-2-7B Reward Model Finetuning

### 问题修复

- 修复了DeepLabV3+ PaddlePaddle模型依赖安装不全的问题。
- 修复了MobileSeg 和Attention U-Net模型训练步骤路径错误问题。
- 修复了GLM模型安装与准备步骤中的路径导引有误的问题。
- 清除了Swin Transformer模型目录下的冗余pycache文件。
- 完善了basicVSR++，basicVSR，RealBasicVSR，TTVSR模型的执行代码示例。

### 版本关联

DeepSparkHub 24.03对应天数软件栈3.1.1版本。

### 贡献者

感谢以下人员做出的贡献：

majorli，songjian，xiaomei.wang，李一力，gaiqin.bai，wenfeng.zhang，吴永乐，sanghui_ilu，shengquan.nian，yanlong.hao，zhaojun0044，李永康，小豆子。

欢迎以任何形式为DeepSparkHub项目贡献。

---

## DeepSparkHub 23.12 Release Notes

### 特性和增强

#### 模型与算法

* 新增了30个算法模型。

<table>
    <tr align="left"><th colspan=6>PyTorch</th></tr>
    <tr>
        <td>ACNet</td>
        <td>Co-DETR</td>
        <td>ControlNet</td>
        <td>DBNet++</td>
        <td>DDPM</td>
        <td>Efficient Conformer V2</td>
    </tr>
    <tr>
        <td>Mask2Former</td>
        <td>Oriented RepPoints</td>
        <td>RepMLP</td>
        <td>RepViT</td>
        <td>RTMDet</td>
        <td>SOLOv2</td>
    </tr>
    <tr align="left"><th colspan=6>PaddlePaddle</th></tr>
    <tr>
        <td>BlazeFace</td>
        <td>ByteTrack</td>
        <td>DLRM</td>
        <td>DQN</td>
        <td>EfficientNetB0</td>
        <td>FastFCN</td>    
    </tr>
    <tr>
        <td>FFM</td>
        <td>GAT</td>
        <td>GraphSAGE</td>
        <td>Pix2Pix</td>
        <td>PP-HumanSegV1</td>
        <td>Res2Net50_14w_8s</td>
    </tr>
    <tr>
        <td>SE_ResNet50_vd</td>
        <td>STDC</td>
        <td>PP-TTS-FastSpeech2</td>
        <td>Xception41</td>
        <td>xDeepFM</td>
        <td>XLNet</td>
     </tr>
</table>

* 新增了基于分布式训练框架的大语言模型训练示例。
    * Megatron-DeepSpeed LLaMA2-7B

### 问题修复

- 修复了YOLOv5模型训练过程中数值类型报错问题。
- 修复了ESRGAN，LIIF，TTSR模型配置文件路径问题。
- 修复了YOLOv3模型安装脚本sudo报错问题。
- 修复了SOLO模型依赖的yapf模块版本问题。
- 修复了YOLOF模型mmcv版本兼容问题。
- 改进了BERT系列模型对本地离线运行的支持。
- 改进了MMDetection工具箱不同版本的安装流程。


### 版本关联

DeepSparkHub 23.12对应天数软件栈3.1.0版本。

### 贡献者

感谢以下人员做出的贡献：

majorli，songjian，may，jun.zhao，wz50525，gaiqin_bai，郝燕龙，巩亚飞，张文风，li.ding，Asltw，吴永乐，westnight，shengquan.nian，la。

欢迎以任何形式为DeepSparkHub项目贡献。

---

## DeepSparkHub 23.09 Release Notes

### 特性和增强

#### 模型与算法

* 新增了30个算法模型。

<table>
    <tr align="left"><th colspan=5>PyTorch</th></tr>
    <tr>
        <td>BYOL</td>
        <td>InternImage</td>
        <td>MobileOne</td>
        <td>MoCoV2</td>
        <td>WSLD</td>
    </tr>
    <tr align="left"><th colspan=5>TensorFlow</th></tr>
    <tr>
        <td>AlexNet</td>
        <td>DeepLabV3-Plus</td>
        <td>FaceNet</td>
        <td>InceptionV3</td>
        <td>VGG</td>
    </tr>
    <tr align="left"><th colspan=5>MindSpore</th></tr>
    <tr>
        <td>CRNN</td>
        <td>InceptionV3</td>
        <td>MobileNetV3</td>
        <td>OpenPose</td>
        <td>ResNeXt50</td>
    </tr>
    <tr align="left"><th colspan=5>PaddlePaddle</th></tr>
    <tr>
        <td>DenseNet121</td>
        <td>GCN</td>
        <td>GPT2-Medium-EN</td>
        <td>HRNet-W32</td>
        <td>MobileNetV3_Large1.0</td>
    </tr>
    <tr>
        <td>MobileSeg</td>
        <td>OCRNet</td>
        <td>PP-ASR-Conformer</td>
        <td>PP-HumanSegV2</td>
        <td>PP-LCNet</td>
    </tr>
    <tr>
        <td>PP-LiteSeg</td>
        <td>PP-PicoDet</td>
        <td>PP-TTS-FastSpeech2</td>
        <td>PP-TTS-HiFiGAN</td>
        <td>ShuffleNetV2</td>
     </tr>
</table>

* 新增了基于分布式训练框架的大语言模型训练示例。
    * Colossal-AI LLaMA-7B
    * DeepSpeed ChatGLM-6B
* 新增了基于深度学习分子动力学套件的水分子模型训练示例。
    * DeePMD-kit water/se_e2_a

### 问题修复

- 修复了Stable Diffusion模型的accelerate包版本问题。
- 修复了PaddlePaddle GCN模型的环境变量设置问题。
- 修复了PyTorch ResNet50模型的多机训练问题。
- 更新了MindSpore DeepLabV3模型VOC数据集链接。
- 新增了DeepSparkHub模型REAME.md文档参考模板。
- 优化了开源工具箱Fairseq/MMDetection/WeNet的模型存放路径和展示方式。

### 版本关联

DeepSparkHub 23.09对应天数软件栈3.1.0版本。

### 贡献者

感谢以下人员做出的贡献：

majorli，zhaojun0044，songjian，gongyafei，gaiqin_bai，Asltw，张文风，吴永乐，丁力，tonychen，牛斯克，la。

欢迎以任何形式为DeepSparkHub项目贡献。

---

## DeepSparkHub 23.06 Release Notes

### 特性和增强

#### 模型与算法

添加了30个基于PyTorch框架的算法模型，新增了网络剪枝、自监督学习、知识蒸馏这3种模型类别。

<table>
    <tr>
        <th rowspan=4>PyTorch</th>
        <td>AlphaPose</td>
        <td>ArcFace</td>
        <td>Attention U-Net</td>
        <td>CBAM</td>
        <td>CosFace</td>
    </tr>
    <tr>
        <td>CspDarknet53</td>
        <td>CWD</td>
        <td>DDRNet</td>
        <td>FaceNet</td>
        <td>FasterNet</td>
    </tr>
    <tr>
        <td>MAE</td>
        <td>Network Slimming</td>
        <td>PointNet++</td>
        <td>RKD</td>
        <td>STDC</td>
    </tr>
    <tr>
        <td>UNet++</td>
        <td>YOLOv6</td>
        <td>YOLOv8</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <th>PyTorch (WeNet)</th>
        <td>Transformer</td>
        <td>U2++ Conformer</td>
        <td>Unified Conformer</td>
        <td></td>
        <td></td>
    </tr>
    <tr>
        <th>PyTorch (MMDetection)</th>
        <td>ATSS</td>
        <td>Cascade R-CNN</td>
        <td>CornerNet</td>
        <td>DCNV2</td>
        <td>RepPoints</td>
    </tr>
    <tr>
        <th>PyTorch (Fairseq)</th>
        <td>BART</td>
        <td>Convolutional</td>
        <td>RoBERTa</td>
        <td>Transformer</td>
        <td></td>
    </tr>
</table>

30个模型中有12个使用了开源工具箱，包括：
- Transformer，U2++ Conformer，Unified Conformer模型基于开源的WeNet工具箱，可以便捷的执行数据集准备、模型训练/测试/导出任务。
- ATSS，Cascade R-CNN，CornerNet，DCNV2，RepPoints模型基于开源的MMDetection工具箱，可以便捷的执行目标检测训练任务。
- BART，Convoluntional，RoBERTa，Transformer模型基于开源的Fairseq工具箱，可以便捷的执行自然语言处理训练任务。

### 问题修复

- 修复了MobileNetV2和YOLOv5模型参数问题。
- 修复了LIIF和VGG16模型Python依赖包缺失问题。
- 修复了Stable Diffusion模型的测试结果图片呈现问题。
- 更新了DLRM模型数据集获取链接。
- 新增了Tacotron2、YOLOv5、SATRN模型的性能指标输出。
- 新增了FairMOT模型的性能指标和精度指标输出。

### 版本关联

DeepSparkHub 23.06对应天数软件栈3.0.0版本。

### 贡献者

感谢以下人员做出的贡献：

majorli，吴永乐，songjian，丁力，shengquan.nian，may，张文风，chenyingtony，yue.chang，westnight，fhfang，li-shi-kun。

欢迎以任何形式为DeepSparkHub项目贡献。

---

## DeepSparkHub 23.03 Release Notes

### 特性和增强

#### 多框架支持
新增了对TensorFlow和MindSpore的支持。

#### 模型与算法
新增了35个算法模型，具体如下。

<table>
    <tr><th colspan="5", align="left">TensorFlow</th></tr>
    <tr>
        <td>BERT</td>
        <td>ResNet50</td>
        <td>SSD</td>
        <td>VNet</td>
        <td>YOLOv3</td>
    </tr>
    <tr><th colspan="5", align="left">MindSpore</th></tr>
    <tr>
        <td>BERT</td>
        <td>DCGAN</td>
        <td>DeepLabV3</td>
        <td>GCN</td>
        <td>SSD</td>
    </tr>
    <tr><th colspan="5", align="left">PyTorch</th></tr>
    <tr>
        <td>BERT NER</td>
        <td>BERT Question<br>Answering</td>
        <td>BERT Text<br>Classification</td>
        <td>BERT Text<br>Summarization</td>
        <td>BEVFormer</td>
    </tr>
    <tr>
        <td>DBNet</td>
        <td>DeepSORT</td>
        <td>FCOS</td>
        <td>PointPillars</td>
        <td>RepVGG</td>
    </tr>
    <tr>
        <td>Retinaface</td>
        <td>SOLO</td>
        <td>Stable Diffusion</td>
        <td>YOLACT++</td>
        <td>YOLOv7</td>
    </tr>
    <tr><th colspan="5", align="left">PaddlePaddle</th></tr>
    <tr>
        <td>BiSeNetV2</td>
        <td>CenterNet</td>
        <td>DeepLabV3+</td>
        <td>DETR</td>
        <td>FCOS</td>
    </tr>
    <tr>
        <td>RepVGG</td>
        <td>ResNeSt50</td>
        <td>RetinaNet</td>
        <td>SOLOv2</td>
        <td>Swin<br>Transformer</td>
    </tr>
</table>

### 问题修复

- 修复了CenterNet模型缺失make.sh文件的问题。
- 更新了ResNet50模型的运行脚本路径。
- 更新了HashNeRF模型的数据集准备和模型执行步骤。

### 版本关联

DeepSparkHub 23.03对应天数软件栈3.0.0版本。

### 贡献者

感谢以下人员做出的贡献：

shengquan.nian，牛冠博，吴永乐，may，majorli6，yue.chang，wenfeng.zhang，丁力，xiaomei.wang，songjian，guanbo，cheneychen2023，Yili。

欢迎以任何形式为DeepSparkHub项目贡献。

---

## DeepSparkHub 22.12 Release Notes

### 特性和增强

- SATRN，Conformer和ngp-nerf模型更新6维度评测数据。
- DLRM和CPM模型增加checkpoint。
- Hashnerf模型增加多卡运行支持。
- 新增PaddlePaddle框架模型19个。

模型名称 | 框架 | 数据集 
-------- | ------ | ----
googlenet | PaddlePaddle | ImageNet
MobileNetV3 | PaddlePaddle | ImageNet
ResNet50 | PaddlePaddle | ImageNet
VGG16 | PaddlePaddle | ImageNet
Mask R-CNN | PaddlePaddle | COCO
SSD | PaddlePaddle | COCO
YOLOv3 | PaddlePaddle | COCO
PP-YOLOE | PaddlePaddle | COCO
PSE | PaddlePaddle | OCR_Recog
CRNN | PaddlePaddle | LMDB
PP-OCR DB | PaddlePaddle | ICDAR2015
DeepLab | PaddlePaddle | COCO
UNet | PaddlePaddle | CityScapes
dnlnet | PaddlePaddle | CityScapes
BERT Pretraining | PaddlePaddle | MNLI
Ernie | PaddlePaddle | corpus
Transformer | PaddlePaddle | wmt14-en-de-pre-processed
DeepFM | PaddlePaddle | Criteo_Terabyte
Wide&Deep | PaddlePaddle | Criteo_Terabyte


### 问题修复

- ssd模型执行prepare.py报错。([#I5Y00S](https://gitee.com/deep-spark/deepsparkhub/issues/I5Y00S))
- vqmivc模型使用数据集问题。([#I63WFR](https://gitee.com/deep-spark/deepsparkhub/issues/I63WFR))
- APCNet模型readme文档缺少数据集准备步骤。([#I63W7K](https://gitee.com/deep-spark/deepsparkhub/issues/I63W7K))
- GCNet模型Readme文档缺少数据集存放位置描述。([#I63W8I](https://gitee.com/deep-spark/deepsparkhub/issues/I63W8I))
- Graph WaveNet模型Readme问题。([#I63WCW](https://gitee.com/deep-spark/deepsparkhub/issues/I63WCW))
- 首页readme文档的clip 模型链接有问题。([#I62HM3](https://gitee.com/deep-spark/deepsparkhub/issues/I62HM3))
- ConvNext,RetinaNet,ACmix readme文档更新下载数据集。([#I6101Y](https://gitee.com/deep-spark/deepsparkhub/issues/I6101Y))
- semantic_segmentation下模型的readme文档需要补充准备数据集步骤。([#I5Y01H](https://gitee.com/deep-spark/deepsparkhub/issues/I5Y01H))
- basicVSR++，basicVSR，LIIF，TTSR，TTVSR模型readme文档问题。([#I63WBL](https://gitee.com/deep-spark/deepsparkhub/issues/I63WBL))
- RNN-T模型执行报错。([#I64SU3](https://gitee.com/deep-spark/deepsparkhub/issues/I64SU3))
- ESRGAN模型执行报错。([#I64SPT](https://gitee.com/deep-spark/deepsparkhub/issues/I64SPT))
- Point-BERT模型运行报错。([#I64SWK](https://gitee.com/deep-spark/deepsparkhub/issues/I64SWK))
- RealBasicVSR模型执行报错。([#I64STQ](https://gitee.com/deep-spark/deepsparkhub/issues/I64STQ))
- HRNet模型运行提示缺少json_tricks等python依赖。([#I63W67](https://gitee.com/deep-spark/deepsparkhub/issues/I63W67))

### 版本关联

DeepSparkHub 22.12对应天数软件栈2.3.0版本。

### 贡献者

感谢以下人员做出的贡献：

majorli，张文风，yufei.chen，jeff guo，李睿，sanghui_ilu，westnight，jianyong.jiang， may，Shikun Li，shengquan.nian，qiangzibro，hui.sang。

欢迎以任何形式为DeepSparkHub项目贡献。
