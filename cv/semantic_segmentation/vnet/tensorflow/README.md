
## Prepare
```
pip3 install -r requirements.txt
```

## Download dataset 
```
python3  download_dataset.py --data_dir ./data
```


## Run training 
### single card
```
python3 examples/vnet_train_and_evaluate.py --gpus 1 --batch_size 8 --base_lr 0.0001 --data_dir ./data/Task04_Hippocampus/ --model_dir ./model_train/
```

### 8 cards
```
python3 examples/vnet_train_and_evaluate.py --gpus 8 --batch_size 8 --base_lr 0.0001 --data_dir ./data/Task04_Hippocampus/ --model_dir ./model_train/

```

## Result

|               | background_dice           |       anterior_dice   | posterior_dice    |
| ---           | ---                       | ---                   | ---               |
|    multi_card |  0.9912699                | 0.83743376            |  0.81537557       |