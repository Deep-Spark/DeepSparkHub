### 1. Install

```
bash clean_megatron-deepspeed.sh
bash build_megatron-deepspeed.sh
bash install_megatron-deepspeed.sh
```

### 2. CI Test

#### 2.1 Test node = 1

```
cd ci && bash run_ci_tests_one_node.sh
```

#### 2.2 Test node >= 2

First, you should make sure something below.

1. The CI Test in 1 node can pass in master node container.
2. Copy master node container environment to other node servers.
3. Make sure the account name, contrainer name is the same in different node servers.
4. Set up password free login between the master node container and other node servers.

Second, set your node server info. You can set up like:

```
## The account in server
export HOST_NAME="username"

## Severs IP, begin with the master node server IP, and split by ","
export ADDR_ARRAY="10.111.222.1,10.111.222.2"

## Container name
export CONTAINER_NAME="megatron-deepspeed"
```

Third, run.

```
cd ci && bash run_ci_tests_multi_node.sh
```

### 3. Run Aquila-7b bf16 pretrain

#### 3.1 Download Dataset

```
bash dataset/download_dataset.sh
```

#### 3.2 Run node=1

```
cd examples/aquila && bash run_aquila_7b_node1_bf16.sh
```

#### 3.3 Run node=2

First, you should make sure something below.

1. The pretrain in 1 node run successfully in master node container.
2. Copy master node container environment to other node servers.
3. Make sure the account name, contrainer name is the same in different node servers.
4. Set up password free login between the master node container and other node servers.
5. Make megatron-deepspeed repo and dataset at same path in different node servers.

Second, set your node server info. You can set up like:

```
## The account in server
export HOST_NAME="username"

## Severs IP, begin with the master node server IP, and split by ","
export ADDR_ARRAY="10.111.222.1,10.111.222.2"

## Container name
export CONTAINER_NAME="megatron-deepspeed"
```

Third, run.

```
cd examples/aquila && bash run_aquila_7b_node2_bf16.sh
```
