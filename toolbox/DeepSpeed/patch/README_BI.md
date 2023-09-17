## build
```shell
bash build_deepspeed.sh
```
## install
```shell
bash install_deepspeed.sh
```
#### 查看编译情况
```shell
ds_report
```

| op name                | installed | compatible |
|------------------------|-----------|------------|
| async_io               | YES       | OKAY       |
| cpu_adagrad            | YES       | OKAY       |
| cpu_adam               | YES       | OKAY       |
| fused_adam             | YES       | OKAY       |
| fused_lamb             | YES       | OKAY       |
| quantizer              | YES       | OKAY       |
| sparse_attn            | YES       | OKAY       |
| spatial_inference      | YES       | OKAY       |
| transformer            | YES       | OKAY       |
| stochastic_transformer | YES       | OKAY       |
| transformer_inference  | YES       | OKAY       |
## clean
```shell
bash clean_deepspeed.sh
```