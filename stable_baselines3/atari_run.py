import os
import sys

import numpy as np
import torch

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.dqn.ddqn import DDQN
from stable_baselines3.dqn.dqn1 import DQN1

sys.path.append("/home/dsi/fiskustal/talfiskus/stable-baselines3/")

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.dqn.ddqn import DDQN
from stable_baselines3.dqn.dqn1 import DQN1
from stable_baselines3.common.callbacks import EvalCallback

import time
from datetime import timedelta

if __name__ == "__main__":
    agent_name = "DQN1" # TODO: change this
    buffer_size = 100000 # 100K
    timesteps = 100000 # Atari 100K
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #SUFT
    tf_lambda = 0
    # v5: disable the built-in frame skip (AtariWrapper already skips 4 frames)
    # and let ALE alone handle sticky actions
    atari_env_kwargs = {"frameskip": 1, "repeat_action_probability": 0.25}
    env = make_atari_env("ALE/Alien-v5", seed=42, env_kwargs=atari_env_kwargs)
    # Stack 4 frames
    env = VecFrameStack(env, n_stack=4)
    # Eval env
    eval_env = make_atari_env("ALE/Alien-v5", seed=42, env_kwargs=atari_env_kwargs)
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_callback = EvalCallback(eval_env, eval_freq=10000, verbose=1, n_eval_episodes=100)
    # select agent by name
    if agent_name == 'DQN':
        model = DQN('CnnPolicy', env, verbose=2, device=device, tf_lambda=tf_lambda, buffer_size=buffer_size)
    elif agent_name == 'DDQN':
        model = DDQN('CnnPolicy', env, verbose=2, device=device, tf_lambda=tf_lambda, buffer_size=buffer_size)
    elif agent_name == 'DQN1':
        model = DQN1('CnnPolicy', env, verbose=2, device=device, tf_lambda=tf_lambda, buffer_size=buffer_size)
    else:
        print("ERROR - no valid agent")

    # Start measure runtime
    start_time = time.time()
    # Train model
    model.learn(total_timesteps=timesteps, callback=eval_callback) #, callback=eval_callback)
    # Record the end time
    end_time = time.time()
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    # print(f"Trained agent mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    # Save model
    # model.save(f"{full_log_dir}/trained_model")
    # plot_results([full_log_dir], timesteps, results_plotter.X_EPISODES, model_name)
    # plt.savefig(f"{full_log_dir}/plot_episodes_reward_plot.png")
    # plot_results([full_log_dir], timesteps, results_plotter.X_TIMESTEPS, model_name)
    # plt.savefig(f"{full_log_dir}/plot_timesteps_reward_plot.png")

    # Calculate the total runtime
    total_time = end_time - start_time
    # Convert total_time to hh:mm:ss format
    formatted_time = str(timedelta(seconds=int(total_time)))
    # Print the total runtime
    print(f"Total train runtime: {formatted_time}")
