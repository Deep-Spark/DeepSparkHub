# DeepSparkHub Release Notes

## DeepSparkHub 23.03 Release Notes

### 特性和增强

#### 多框架支持
新增了对TensorFlow和MindSpore的支持。

#### 模型与算法
新增了35个算法模型，具体如下。

**TensorFlow**

BERT | ResNet50 | SSD | YOLOv3 | VNet
-----| -------- | --- | ------ | ----

**MindSpore**

BERT | SSD | DCGAN | DeepLabV3 | GCN
-----| --- | ----- | --------- | ---

**PaddlePaddle**

Swin Transformer | RepVGG | ResNeSt50 | DeepLabV3+ | BiSeNetV2
-----------------| ------ | --------- | ---------- | ---------

RetinaNet        | FCOS   | DETR      | CenterNet  | SOLOv2
-----------------| ------ | --------- | ---------- | ---------

**PyTorch**

RepVGG           | PointPillars | BEVFormer  | YOLACT++ | BERT Question Answering
-----------------| ------       | ---------  | -------- | ---------

YOLOv7           | FCOS         | Retinaface | DeepSORT | BERT Text Classification
-----------------| ------       | ---------  | -------- | ---------

Stable Diffusion | DBNet        | SOLO       | BERT NER | BERT Text Summarization
-----------------| ------       | ---------  | -------- | ---------

### 问题修复

- 更新了ResNet50模型的运行脚本路径。
- 更新了HashNeRF模型的数据集准备和模型执行步骤。

### 贡献者

感谢以下人员做出的贡献：

shengquannian，吴永乐，may，牛斯克，张文风，la，丁力，song.jian，westnight，
cheneychen2023，郝燕龙，tonychen，Shikun Li，majorli。

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

### 贡献者

感谢以下人员做出的贡献：

majorli，张文风，yufei.chen，jeff guo，李睿，sanghui_ilu，westnight，jianyong.jiang，
may，Shikun Li，shengquan.nian，qiangzibro，hui.sang。

欢迎以任何形式为DeepSparkHub项目贡献。
