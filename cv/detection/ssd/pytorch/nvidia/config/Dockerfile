ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.06-py3
FROM ${FROM_IMAGE_NAME}

# Install dependencies for system configuration logger
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        infiniband-diags \
        pciutils \
 && rm -rf /var/lib/apt/lists/*

# Configure environment variables
ENV OMP_NUM_THREADS=1
ENV OPENCV_FOR_THREADS_NUM=1

