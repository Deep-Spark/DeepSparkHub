# 单机训练
    bash ./ds_train_finetune.sh


# 多机训练：
## 启动脚本
    在 host 机器上调用以下脚本即可：
    bash ./ds_train_finetune_multi_nodes.sh

## 注意事项
1. 起容器时使用 --network=host

2. 非 host 机器需要与 host 之间设置ssh免密互连

    设置方式：
    2.1 多机上分别设置 sshd 端口监听端口
    /usr/sbin/sshd
    /usr/sbin/sshd -p 12345

    如果未安装sshd，需要先安装sshd

    2.2 设置公钥认证
    cd ~/.ssh
    ssh-keygen -t rsa
    ssh-copy-id -i ~/.ssh/id_rsa.pub root@10.113.2.103 -p 12345 （所有非host和host上执行此语句，将公钥写入host ssh 配置中，使host能免密登录host自身和所有非host机器上）

3. ds_train_finetune_multi_nodes.sh 中 MASTER_ADDR, MASTER_ADDR 根据实际环境设置，其中 MASTER_PORT 为torch用来通信的，须与ssh远程登录的端口不要重复。

4. 在本目录中使用hostfile，其中 nv_103 和 nv_104 与 ~/.ssh/config（下一步添加） 中 host 对应。

5. 在环境中添加文件 ~/.ssh/config,格式如下：
    host nv_103
        HostName 10.113.2.103
        Port 12345
    host nv_104
        HostName 10.113.2.104
        Port 12345

    其中 nv_103，nv_104 与hostfile中host对应；
         HostName 分别为多机的ip；
         Port 是供deepspeed使用，用于ssh远程登录。

## 其他说明
1. 如果遇到环境变量导致的问题可进行如下解决，如：
    1.1. pip command not found
    添加文件 ~/.deepspeed_env，在其中添加环境变量PATH在真正训练的目标机上的值，如：PATH=/opt/conda/bin:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/tensorrt/bin

2. 使用 deepspeed_no_cpu_offload.json 和 deepspeed.json 可以在32G 以下的 8*GPU 上进行双机训练

# 源码出处
    https://github.com/THUDM/ChatGLM-6B.git