# DeepSparkHub容器镜像构建说明

本文档旨在协助社区用户在本地构建出在天数智芯加速卡上运行的DeepSparkHub容器镜像，以运行仓库中的模型。

## 系统要求
- x86_64
- Linux
- Docker
- 已安装驱动

## 构建前准备

### 获取离线安装包

1. 访问[CUDA Toolkit 10.2 Download](https://developer.nvidia.com/cuda-10.2-download-archive)页面获取Linux版CUDA离线安装包。

2. 访问[天数智芯官网-资源中心](https://support.iluvatar.com/#/DocumentCentre?id=1&nameCenter=2&productId=381380977957597184)页面获取Linux版软件栈离线安装包：
    - 如已有账号，则点击右上角“登录”按钮进行下载。
    - 如无账号，则点击右上角“登录”按钮后选择“去注册”申请账号后下载。

3. 下载好的离线安装包解压后放入docker/Iluvatar目录。

### 确认本地构建目录

软件栈版本对应关系可参看[DeepSparkHub版本日志](../../RELEASE.md#版本关联)。离线安装包解压放置好后的目录如下所示：

```bash
docker/Iluvatar
├── 3.7 ## Python3.7 版本whl包，离线包已包含，无需另外下载
├── corex-installer-***.run ## 软件栈离线安装包
├── cuda_***_linux.run ## CUDA离线安装包
└── Dockerfile ## 镜像构建模板文件
```

## 构建镜像

**设置deepsparkhub信息：**
```bash
## 设置镜像名称
$ IMAGE_NAME=deepsparkhub
## 设置镜像版本，与DeepSparkHub发布版本号对应 
$ IMAGE_VERSION=23.06
```

**设置离线安装包名称：**
```bash
$ COREX_INSTALL=corex-installer-***.run
$ CUDA_INSTALL=cuda_***_linux.run
```

**构建容器镜像：**
```bash
$ docker build --build-arg IMAGE_VERSION=${IMAGE_VERSION} \
               --build-arg CUDA_INSTALL=${CUDA_INSTALL} \
               --build-arg COREX_INSTALL=${COREX_INSTALL} \
               -t ${IMAGE_NAME}:${IMAGE_VERSION} .
```

**确认镜像构建成功：**
```bash
$ docker images | grep ${IMAGE_NAME}
```

**运行容器：**
```bash
## 设置容器名称
$ DOCKER_NAME=deepsparkhub_test
## 启动容器
$ sudo docker run -itd --name ${DOCKER_NAME} \
                  --privileged --cap-add=ALL \
                  --ipc=host --pid=host ${IMAGE_NAME}:${IMAGE_VERSION}
## 登录容器
$ docker exec -it ${DOCKER_NAME} bash
## 初始目录为 /root/deepsparkhub
$ pwd
/root/deepsparkhub
```

**模型训练：**

以PyTorch版的ResNet50模型为例，如需运行其他模型，参考对应模型目录的README文档。

```bash
## 切换到ResNet50的PyTorch模型目录
$ cd cv/classification/resnet50/pytorch/
## 查看模型README文档
$ cat README.md
## 下载好ImageNet数据集后，指定路径并运行ResNet50模型
$ PATH_TO_IMAGENET=/path/to/imagenet
$ export CUDA_VISIBLE_DEVICES=0
$ bash scripts/fp32_1card.sh --data-path ${PATH_TO_IMAGENET}
```
