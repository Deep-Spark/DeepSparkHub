# Llama2-34B (Megatron-DeepSpeed)

## Model Description

Llama2-34B is a powerful large language model developed by Meta, featuring 34 billion parameters. It builds upon the
success of its predecessors with a 40% larger training corpus and an extended context length of 4096 tokens, enabling
better understanding of longer texts. This model excels in various natural language tasks, including text generation,
comprehension, and dialogue. Its enhanced architecture and training methodology make it a versatile tool for AI
applications, offering state-of-the-art performance in language understanding and generation.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V150 | 3.4.0     |  24.06  |

## Model Preparation

### Configure 4-node environment

1. Configure the same runing environment on each node and make sure the docker container names are the same
2. Set ssh non-encryption connection on docker container:

```sh
# a. Generate the secret key on master node:
ssh-keygen

# b. Copy the public key to other nodes:
ssh-copy-id -i ~/.ssh/id_rsa.pub ${host_name}  ## {host_name} can be a specified Ip address or domain name
```

### Installation on all nodes

```sh
# install
cd <DeepSparkHub_Root>/toolbox/Megatron-DeepSpeed
```

### Preparing datasets on all nodes

```sh
cd dataset
mkdir BookCorpusDataset && cd BookCorpusDataset
wget https://the-eye.eu/public/AI/pile_neox/data/BookCorpusDataset_text_document.bin
wget https://the-eye.eu/public/AI/pile_neox/data/BookCorpusDataset_text_document.idx
```

## Model Training

Execute the following command on master node.

```sh
cd examples/llama2/
```

1. Modify run_meg_llama2_34b_node4.sh according to your machine: for example, HOST_NAME, ADDR_ARRAY, CONTAINER_NAME,
   NCCL_SOCKET_IFNAME

2. executing run_meg_llama2_34b_node4.sh

```sh
bash run_meg_llama2_34b_node4.sh
```

a. If there is an error: Got permission denied while trying to connect to the Docker daemon socket at
unix:///var/ru，you can execute the following command on all nodes:

```sh
usermod -aG docker ${user_name} 
systemctl restart docker
chmod 666 /var/run/docker.sock

```

b. If an error occurs that the dataset file does not exist,You can copy the dataset file to other nodes by executing the
following command:

```sh
scp -r ../../dataset/gpt_small_117M/gpt_small_117M_text_document ${user_name}@${host_name}:path/to/megatron-deepspeed/dataset/gpt_small_117M/gpt_small_117M_text_document
```

## References

- [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed)
