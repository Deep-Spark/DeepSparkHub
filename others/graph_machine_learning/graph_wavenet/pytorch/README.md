# Graph WaveNet

## Model description

Graph WaveNet is a graph neural network designed for spatial-temporal graph
modeling. It captures both spatial dependencies and long-range temporal patterns
using two key innovations: an adaptive dependency matrix for spatial
relationships and stacked dilated 1D convolutions for long-term temporal
dependencies. These components are integrated into a unified, end-to-end
framework. Graph WaveNet effectively handles complex, large-scale datasets,
excelling in tasks like traffic prediction and energy forecasting. Experimental
results on datasets like METR-LA and PEMS-BAY show its superior performance
compared to existing models in terms of accuracy and efficiency.

## Step 1: Installing packages

```shell
pip3 install -r requirements.txt
```

## Step 2: Preparing datasets

### Step 2.1: Download METR-LA and PEMS-BAY data from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) links provided by [DCRNN](https://github.com/liyaguang/DCRNN)

### Step 2.2: Process raw data

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

## Step 3: Training

```shell
python3 train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj
```
