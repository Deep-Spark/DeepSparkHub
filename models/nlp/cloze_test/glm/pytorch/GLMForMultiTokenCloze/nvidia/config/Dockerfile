FROM nvcr.io/nvidia/pytorch:21.12-py3

RUN /bin/bash -c "pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple"

RUN /bin/bash -c "pip install h5py"

RUN /bin/bash -c "uname -a"
RUN /bin/bash -c alias python3=python

RUN if [[ "$PYTORCH_BUILD_VERSION" == 1.8* ]]; then \
        pip uninstall -y apex; \
    fi
