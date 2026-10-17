import sys


sys.path.append("/home/dsi/fiskustal/talfiskus/stable-baselines3-new/stable-baseline3/")

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, TD3, DDPG
from stable_baselines3.ddpg.ddpg_sl import DDPGSL
from stable_baselines3.sac.sac_sl import SACSL
from stable_baselines3.td3.td3_sl import TD3SL
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise

import time
from datetime import timedelta

if __name__ == "__main__":
    # Usage:
    # python mujoco_cf.py <device> <tf_lambda> <agent_name> <timesteps> <run_number> <gym_env_name> <buffer_size>
    #                     <batch_size> <train_freq>
    # for the short-long agents (DDPGSL, TD3SL, SACSL) <buffer_size> is the long term buffer size, followed by:
    # <short_buffer_size> <sampling_fraction>
    # get the device from command line
    run_device = sys.argv[1]
    print("device", run_device)
    # get CF lambda from command line
    tf_lambda = float(sys.argv[2])
    print("tf_lambda", tf_lambda)
    # get the agent from command line
    agent_name = sys.argv[3]
    print("agent_name", agent_name)
    # get the timestep from command line
    timesteps = int(sys.argv[4])
    print("timesteps", timesteps)
    # get the run number from command line
    run_number = sys.argv[5]
    print("run_number", run_number)
    # get the environment from command line
    gym_env_name = sys.argv[6]
    print("gym_env_name", gym_env_name)
    # get the buffer size from command line
    buffer_size = int(sys.argv[7])
    print("buffer_size", buffer_size)
    # get the minibatch size from command line
    batch_size = int(sys.argv[8])
    print("batch_size", batch_size)
    # get the train frequency from command line: one gradient update every train_freq actions
    # (replay ratio 1 / train_freq)
    train_freq = int(sys.argv[9])
    print("train_freq", train_freq)
    env = gym.make(gym_env_name, render_mode=None)
    eval_env = gym.make(gym_env_name, render_mode=None)
    eval_callback = EvalCallback(eval_env, eval_freq=100000, verbose=1, n_eval_episodes=100, deterministic=False)

    # RL Baselines3 Zoo hyperparameters for the MuJoCo envs
    # https://github.com/DLR-RM/rl-baselines3-zoo/tree/master/hyperparams
    n_actions = env.action_space.shape[0]
    td3_ddpg_kwargs = dict(
        learning_starts=10000,
        batch_size=batch_size,
        train_freq=train_freq,
        action_noise=NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)),
    )
    sac_kwargs = dict(learning_starts=10000, batch_size=batch_size, train_freq=train_freq)

    if agent_name == 'DDPG':
        model = DDPG('MlpPolicy', env, verbose=2, device=run_device, buffer_size=buffer_size, **td3_ddpg_kwargs)
    elif agent_name == 'TD3':
        model = TD3('MlpPolicy', env, verbose=2, device=run_device, buffer_size=buffer_size, **td3_ddpg_kwargs)
    elif agent_name == 'SAC':
        model = SAC('MlpPolicy', env, verbose=2, device=run_device, buffer_size=buffer_size, tf_lambda=tf_lambda,
                    **sac_kwargs)
    elif agent_name in ('DDPGSL', 'TD3SL', 'SACSL'):
        short_buffer_size = int(sys.argv[10])
        print("short_buffer_size", short_buffer_size)
        sampling_fraction = float(sys.argv[11])
        print("sampling_fraction", sampling_fraction)
        sl_agent = {'DDPGSL': DDPGSL, 'TD3SL': TD3SL, 'SACSL': SACSL}[agent_name]
        sl_kwargs = sac_kwargs if agent_name == 'SACSL' else td3_ddpg_kwargs
        model = sl_agent('MlpPolicy', env, verbose=2, device=run_device, tf_lambda=tf_lambda, buffer_size=short_buffer_size,
                         long_term_buffer_size=buffer_size, sampling_fraction=sampling_fraction, **sl_kwargs)
    else:
        print("ERROR - no valid agent")
    # Set new logger
    # model.set_logger(new_logger)
    # Start measure runtime
    start_time = time.time()
    # Train model
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    # Record the end time
    end_time = time.time()

    # Calculate the total runtime
    total_time = end_time - start_time
    # Convert total_time to hh:mm:ss format
    formatted_time = str(timedelta(seconds=int(total_time)))
    # Print the total runtime
    print(f"Total train runtime: {formatted_time}")
