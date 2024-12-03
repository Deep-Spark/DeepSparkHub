# KAN

## Model description
Kolmogorov-Arnold Networks (KANs) are promising alternatives of Multi-Layer Perceptrons (MLPs). KANs have strong mathematical foundations just like MLPs: MLPs are based on the universal approximation theorem, while KANs are based on Kolmogorov-Arnold representation theorem. KANs and MLPs are dual: KANs have activation functions on edges, while MLPs have activation functions on nodes. This simple change makes KANs better (sometimes much better!) than MLPs in terms of both model accuracy and interpretability. 


## Run
```shell
$ bash ./run_train.sh

```

## Result
| Model       | Training speed   |
|-------------|------------------|
| KAN         | 6490 samples/sec |


## Reference

- [pykan](https://github.com/KindXiaoming/pykan/tree/master)

