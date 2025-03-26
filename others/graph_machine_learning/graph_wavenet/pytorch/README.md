# Graph WaveNet

## Model Description

Graph WaveNet is an advanced neural network architecture designed for spatial-temporal graph modeling, excelling in
tasks like traffic prediction and energy forecasting. It combines adaptive dependency matrices for capturing spatial
relationships with stacked dilated 1D convolutions for long-term temporal dependencies. This unified framework
effectively handles complex, large-scale datasets, demonstrating superior performance in accuracy and efficiency. Graph
WaveNet's innovative approach to modeling both spatial and temporal dependencies makes it a powerful tool for analyzing
and predicting patterns in dynamic, interconnected systems.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 2.2.0     |  22.09  |

## Model Preparation

### Prepare Resources

Download METR-LA and PEMS-BAY data from [Google
Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu
Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) links provided by [DCRNN](https://github.com/liyaguang/DCRNN)

Process raw data.

```shell
# Create data and garage directories
mkdir -p data/
mkdir -p garage/

# Get adj_mx_bay.pkl and adj_mx.pkl
mkdir -p data/sensor_graph/
wget https://github.com/liyaguang/DCRNN/raw/master/data/sensor_graph/adj_mx_bay.pkl -P data/sensor_graph/
wget https://github.com/liyaguang/DCRNN/raw/master/data/sensor_graph/adj_mx.pkl -P data/sensor_graph/

# Put METR-LA and PEMS-BAY data in data/
data/
├── metr-la.h5
├── pems-bay.h5
└── sensor_graph
    ├── adj_mx_bay.pkl
    └── adj_mx.pkl

# METR-LA
python3 generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python3 generate_training_data.py --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5

```

### Install Dependencies

```shell
pip3 install -r requirements.txt
```

## Model Training

```shell
python3 train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj
```
