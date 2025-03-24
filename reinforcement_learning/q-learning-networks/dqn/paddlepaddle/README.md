# DQN

## Model Description

DQN (Deep Q-Network) is a foundational reinforcement learning algorithm that combines Q-Learning with deep neural
networks. As a value-based method, it uses a critic network to estimate action quality in high-dimensional state spaces.
DQN introduces experience replay and target network stabilization to enable stable training. This approach
revolutionized AI capabilities in complex environments, achieving human-level performance in Atari games and forming the
basis for advanced decision-making systems in robotics and game AI.

## Model Preparation

### Install Dependencies

```bash
git clone  https://github.com/PaddlePaddle/PARL.git
cd PARL/examples/DQN
pip3 install -r requirements.txt
pip3 install matplotlib
pip3 install urllib3==1.26.6
```

## Model Training

```bash
# 1 GPU Training
python3 train.py

# Evaluation
mv ../../../evaluate.py ./
python3 evaluate.py
```

## Model Results

Performance of DQN playing CartPole-v0.

| Model | GPU     | Reward |
|-------|---------|--------|
| DQN   | BI-V100 | 200.0  |

## Reference

- [PARL](https://github.com/PaddlePaddle/PARL)
- [Paper](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)
