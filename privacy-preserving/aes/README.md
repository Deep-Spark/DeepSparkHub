# AES

## Model description

A typical use case for AES is to encrypt\decrypt some data. In tradition the AES algorithm always run on the CPU platform and get a bad performance.So some companies and organizations had done some parallel scheme to improve performance under AES algorithm. Iluvatar as a GPGPU chip company is also focus on the AES algorithm performance. We shared the GPU code which work with BI100 GPU can get a better performance on AES encrypt and decrypt. it will get at least 70 times improvement the CPU platform(Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz, 128 processor).


## Build

You can build the demo use clang++ whit the cmd as below:

```bash
clang++ iluvatar-gpu-aes.cu -o gpuaes -L /usr/local/corex/lib64/ -lcudart
```

## Run

You can run the gpuaes demo with 

```bash
./gpuaes 
```

## Output

It will output some debug message as below, and you can get the performance from the message.

"
plaintext numbytes is 102800000 Bytes
GPU encrypt used time:0.380751 sec, and save the ciphertext to ciphertext.txt
GPU decrypt used time:0.552250 sec,  and save the plaintext to ciphertext-decrypt.txt
GPU run AES cost:0.933001 sec
"

## Performance Data

We have done the comparative test between the CPU and Iluvatar BI-V100 GPU. Test data shown as below:

```

| **File Size(Bytes)** | **CPU Total Time(s)** | **GPU Total Time(s)** | **CPU Encrypt time(s)** | **GPU Encrypt time(s)** | **CPU Decrypt Time(s)** | **GPU Decrypt Time(s)** |
| :------------------: | :-------------------: | :-------------------: | :---------------------: | :---------------------: | :---------------------: | :---------------------: |
|         1028         |       0.002252        |       0.002662        |         0.00093         |        0.002233         |        0.001322         |        0.000429         |
|        10280         |       0.019031        |        0.00373        |        0.008991         |        0.003121         |         0.01004         |        0.000609         |
|        20560         |       0.032517        |       0.003651        |        0.017611         |        0.002946         |        0.014906         |        0.000705         |
|        30840         |       0.040336        |       0.008543        |        0.023546         |        0.002193         |         0.01679         |         0.00635         |
|        41120         |       0.047423        |       0.005174        |        0.028465         |        0.004259         |        0.018958         |        0.000915         |
|        51400         |       0.057401        |       0.003809        |        0.033969         |        0.003112         |        0.023432         |        0.000697         |
|        61680         |       0.081131        |        0.00341        |        0.035402         |        0.002665         |        0.045729         |        0.000745         |
|        71960         |        0.07334        |       0.003341        |        0.041557         |         0.00241         |        0.031783         |        0.000931         |
|        82240         |       0.073693        |       0.003952        |        0.042604         |        0.002792         |        0.031089         |         0.00116         |
|        92520         |       0.069063        |       0.004811        |         0.03111         |        0.003305         |        0.037953         |        0.001506         |
|        102800        |       0.086009        |       0.006112        |         0.04673         |        0.004459         |        0.039279         |        0.001653         |
|        205600        |        0.17293        |       0.004879        |        0.073755         |        0.003202         |        0.099175         |        0.001677         |
|        308400        |       0.257353        |       0.006376        |        0.106697         |        0.004139         |        0.150656         |        0.002237         |
|        411200        |        0.28778        |       0.011421        |        0.131373         |        0.008021         |        0.156407         |         0.0034          |
|        514000        |       0.414744        |       0.009324        |        0.171094         |        0.005831         |         0.24365         |        0.003493         |
|        616800        |       0.449304        |       0.010414        |        0.215339         |        0.005563         |        0.233965         |        0.004851         |
|        719600        |       0.516401        |       0.010779        |        0.229566         |        0.006011         |        0.286835         |        0.004768         |
|        822400        |       0.584663        |       0.011307        |        0.259756         |        0.005742         |        0.324907         |        0.005565         |
|        925200        |       0.674667        |       0.012487        |        0.293303         |        0.006823         |        0.381364         |        0.005664         |
|       1028000        |       0.751519        |        0.01351        |        0.332662         |         0.00651         |        0.418857         |          0.007          |
|       2056000        |       1.446726        |       0.023609        |        0.609177         |        0.011234         |        0.837549         |        0.012375         |
|       3084000        |       2.196561        |       0.031345        |        0.925097         |        0.013653         |        1.271464         |        0.017692         |
|       4112000        |       2.940632        |       0.040346        |        1.204424         |        0.017288         |        1.736208         |        0.023058         |
|       5140000        |       3.672653        |       0.051222        |        1.533855         |        0.022738         |        2.138798         |        0.028484         |
|       6168000        |       4.360379        |       0.058291        |        1.818599         |        0.023309         |         2.54178         |        0.034982         |
|       7196000        |       5.102705        |       0.067794        |        2.179289         |        0.027286         |        2.923416         |        0.040508         |
```


## Questions and Comments

If you have any questions or comments, please drop a line to tianyuan.zhang@iluvatar.ai. We look forward to your feedback.
