# Copyright (c) 2023, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

FROM ubuntu:22.04

LABEL image=deepspark/deepsparkhub

ARG IMAGE_VERSION=23.03
ARG PYTHON_VERSION=3.7
ARG CUDA_INSTALL
ARG COREX_INSTALL

USER root

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /root/

RUN sed -i '/^deb.*secur/d; s@/archive.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g' /etc/apt/sources.list

## Install dependancies
RUN apt clean && apt update && apt install dialog -y && \
    apt install -y -qq --no-install-recommends unzip build-essential cmake ca-certificates \
                    make git curl vim wget \
                    perl lsb-release \
                    iproute2 pciutils \
                    kmod openssh-server \
                    automake libglib2.0-0 \
                    autoconf m4 libncursesw5 \
                    libbz2-dev \
                    cmake libgl1-mesa-glx python3-distutils && rm -rf /var/lib/apt/lists/*

## Install python 3.7
# RUN curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3 get-pip.py
RUN apt-get update && apt-get install -y build-essential zlib1g-dev libncurses5-dev liblzma-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget

RUN wget https://www.python.org/ftp/python/3.7.11/Python-3.7.11.tgz && tar -zxvf Python-*.tgz && rm -f Python-*.tgz && \
(cd Python-* && ./configure --enable-optimizations --enable-shared && make altinstall) && rm -rf Python-* && \
(cd /usr/local/bin/ && ln -sf python3.7 python && ln -sf python3.7 python3) && \
echo /usr/local/lib64 > /etc/ld.so.conf.d/local.conf && ldconfig

## Install CUDA
COPY ${CUDA_INSTALL} .
RUN sh ${CUDA_INSTALL} -- --silent --toolkit --librarypath=/usr/local/cuda --override
RUN rm ${CUDA_INSTALL}

## Install CoreX
COPY ${COREX_INSTALL} .
RUN bash ${COREX_INSTALL} --silent --toolkit
RUN rm ${COREX_INSTALL} 

## Install python library
COPY ${PYTHON_VERSION}/ ./${PYTHON_VERSION}

RUN curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3 get-pip.py && \
    pip3 install ${PYTHON_VERSION}/*.whl && pip3 install ${PYTHON_VERSION}/paddle/*.whl && \
    pip3 install ${PYTHON_VERSION}/mindspore/*.whl

RUN rm -rf get-pip.py ./${PYTHON_VERSION}

## Clone deepspark
RUN git clone --depth 1 --branch ${IMAGE_VERSION} https://gitee.com/deep-spark/deepsparkhub.git

WORKDIR /root/deepsparkhub

RUN echo 'export PATH=/usr/local/corex/bin:$PATH' >> ~/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

### CMD ["/usr/bin/bash"]
