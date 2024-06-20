# Llama2-34B (Megatron-DeepSpeed)

## Model description

Llama 2 is a large language model released by Meta in 2023, with parameters ranging from 7B to 70B. Compared to LLaMA, the training corpus of Llama 2 is 40% longer, and the context length has been upgraded from 2048 to 4096, allowing for understanding and generating longer texts.

## Step1: Configure 4-node environment

1. Configure the same runing environment on each node and make sure the docker container names are the same
2. Set ssh non-encryption connection on docker container:

```bash
# a. Generate the secret key on master node:
ssh-keygen

# b. Copy the public key to other nodes:
ssh-copy-id -i ~/.ssh/id_rsa.pub ${host_name}  ## {host_name} can be a specified Ip address or domain name
```

## Step 2: Installation on all nodes

```bash
# Clone
git clone https://gitee.com/deep-spark/Megatron-DeepSpeed.git
cd Megatron-DeepSpeed/
# Install
bash build_megatron-deepspeed.sh && bash install_megatron-deepspeed.sh
pip3 install urllib3==1.23
```

## Step 3: Preparing datasets on all nodes

```bash
cd dataset
mkdir BookCorpusDataset && cd BookCorpusDataset
wget https://the-eye.eu/public/AI/pile_neox/data/BookCorpusDataset_text_document.bin
wget https://the-eye.eu/public/AI/pile_neox/data/BookCorpusDataset_text_document.idx
```

## Step 4: Training by executing the following command on master node

```bash
cd examples/llama2/
```

1. Modify run_meg_llama2_34b_node4.sh according to your machine: for example, HOST_NAME, ADDR_ARRAY, CONTAINER_NAME, NCCL_SOCKET_IFNAME

2. executing run_meg_llama2_34b_node4.sh

```bash
bash run_meg_llama2_34b_node4.sh
```

a. If there is an error: Got permission denied while trying to connect to the Docker daemon socket at unix:///var/ru，you can execute the following command on all nodes:

```bash
usermod -aG docker ${user_name} 
systemctl restart docker
chmod 666 /var/run/docker.sock

```

b. If an error occurs that the dataset file does not exist,You can copy the dataset file to other nodes by executing the following command:

```bash
scp -r ../../dataset/gpt_small_117M/gpt_small_117M_text_document ${user_name}@${host_name}:path/to/megatron-deepspeed/dataset/gpt_small_117M/gpt_small_117M_text_document
```

## Results

|  GPUs   |              Model              | Training speed |
| :-----: | :-----------------------------: | :------------: |
| BI-V150 | Llama2-34B (Megatron-DeepSpeed) |                |

## Reference

- [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed)
