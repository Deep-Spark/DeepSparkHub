# Copyright (c) 2023, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
import os
import gym
import numpy as np
import parl
from parl.utils import logger, ReplayMemory
from cartpole_model import CartpoleModel
from cartpole_agent import CartpoleAgent
from parl.env import CompatWrapper, is_gym_version_ge
from parl.algorithms import DQN
from matplotlib import animation
import matplotlib.pyplot as plt


MEMORY_SIZE = 200000
LEARNING_RATE = 0.0005
GAMMA = 0.99
def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
 
    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
 
    patch = plt.imshow(frames[0])
    plt.axis('off')
 
    def animate(i):
        patch.set_data(frames[i])
 
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)
    
# evaluate 5 episodes
def run_evaluate_episodes(agent, eval_episodes=5, render=False):
    # Compatible for different versions of gym
    if is_gym_version_ge("0.26.0") and render:  # if gym version >= 0.26.0
        env = gym.make('CartPole-v1', render_mode="rgb_array")
    else:
        env = gym.make('CartPole-v1')
    env = CompatWrapper(env)

    eval_reward = []
    
    frames = []
    for i in range(eval_episodes):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                frames.append(env.render())
            if done:
                break
        
        save_frames_as_gif(frames)
        eval_reward.append(episode_reward)
        
    return np.mean(eval_reward)

def main():
    env = gym.make('CartPole-v0')
    # Compatible for different versions of gym
    env = CompatWrapper(env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    # set action_shape = 0 while in discrete control environment
    rpm = ReplayMemory(MEMORY_SIZE, obs_dim, 0)

    # build an agent
    model = CartpoleModel(obs_dim=obs_dim, act_dim=act_dim)
    alg = DQN(model, gamma=GAMMA, lr=LEARNING_RATE)
    agent = CartpoleAgent(
        alg, act_dim=act_dim, e_greed=0.1, e_greed_decrement=1e-6)

    if os.path.exists('./model.ckpt'):
            env_eposide =5 
            agent.restore('./model.ckpt')
            eval_reward = run_evaluate_episodes(agent, env_eposide, render=True)
            logger.info('eval_reward:{}'.format(eval_reward))
            exit()
        
if __name__ == '__main__':
    main()
