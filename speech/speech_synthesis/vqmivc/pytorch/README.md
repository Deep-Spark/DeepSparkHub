# vqmivc

## Model description

One-shot voice conversion (VC), which performs conversion across arbitrary speakers with only a single target-speaker utterance for reference, can be effectively achieved by speech representation disentanglement. Existing work generally ignores the correlation between different speech representations during training, which causes leakage of content information into the speaker representation and thus degrades VC performance. To alleviate this issue, we employ vector quantization (VQ) for content encoding and introduce mutual information (MI) as the correlation metric during training, to achieve proper disentanglement of content, speaker and pitch representations, by reducing their inter-dependencies in an unsupervised manner. Experimental results reflect the superiority of the proposed method in learning effective disentangled speech representations for retaining source linguistic content and intonation variations, while capturing target speaker characteristics. In doing so, the proposed approach achieves higher speech naturalness and speaker similarity than current state-of-the-art one-shot VC systems. Our code, pre-trained models and demo are available at https://github.com/Wendison/VQMIVC.


## Step 1: Preparing datasets

```shell
mkdir -p /home/data/vqmivc/
cd /home/data/vqmivc/
wget https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip
unzip VCTK-Corpus-0.92.zip
```

## Step 2: Preprocess

```shell
cd ${DEEPSPARKHUB_ROOT}/speech/speech_synthesis/vqmivc/pytorch/
pip3 install -r requirements_bi.txt
ln -s /home/data/vqmivc/data data
python3 preprocess.py
```

## Step 3: Training

* Training with mutual information minimization (MIM):

```shell
python3 train.py use_CSMI=True use_CPMI=True use_PSMI=True
```

* Training without MIM:

```shell
python3 train.py use_CSMI=False use_CPMI=False use_PSMI=False 
```

## Results on BI-V100

| Card Type        | recon loss   |  cps loss  | vq loss | perpexlity | lld cs loss | mi cs loss | lld ps loss | mi ps loss | lld cp loss | mi cp loss |used time(s)|
| --------   | -----:  | :----:  | :----:  | :----:  | :----:  | :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| BI      |0.635|1.062 |0.453 |401.693 |110.958|2.653E-4|0.052|0.001|219.895|0.021|4.315|

## Reference
https://github.com/Wendison/VQMIVC
