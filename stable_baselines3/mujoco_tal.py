import sys


sys.path.append("/home/dsi/fiskustal/talfiskus/stable-baselines3/")

import gymnasium as gym
from stable_baselines3 import SAC, TD3, DDPG
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise

import numpy as np

import time
from datetime import timedelta

if __name__ == "__main__":
    # There already exists an environment generator
    # that will make and wrap atari environments correctly.
    # Here we are also multi-worker training (n_envs=4 => 4 environments)
    # get the device
    run_device = "cpu"
    print("device", run_device)
    # get CF lambda from command line
    tf_lambda = 0.0
    print("tf_lambda", tf_lambda)
    # get the agent
    agent_name = "TD3"
    print("agent_name", agent_name)
    # get the timestep from command line
    timesteps = 1000000
    print("timesteps", timesteps)
    # get the run number from command line
    run_number = 1
    print("run_number", run_number)
    # get the environment from command line
    gym_env_name = "Ant-v5"
    print("gym_env_name", gym_env_name)
    buffer_size = 1000000
    print("buffer_size", buffer_size)
    env = gym.make(gym_env_name, render_mode=None)
    eval_env = gym.make(gym_env_name, render_mode=None)
    eval_callback = EvalCallback(eval_env, eval_freq=100000, verbose=1, n_eval_episodes=100, deterministic=False)

    # RL Baselines3 Zoo hyperparameters for the MuJoCo envs
    # https://github.com/DLR-RM/rl-baselines3-zoo/tree/master/hyperparams
    n_actions = env.action_space.shape[0]
    td3_ddpg_kwargs = dict(
        learning_starts=10000,
        action_noise=NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)),
    )
    sac_kwargs = dict(learning_starts=10000)

    if agent_name == 'DDPG':
        model = DDPG('MlpPolicy', env, verbose=2, device=run_device, buffer_size=buffer_size, tf_lambda=tf_lambda, **td3_ddpg_kwargs)
    elif agent_name == 'TD3':
        model = TD3('MlpPolicy', env, verbose=2, device=run_device, buffer_size=buffer_size, tf_lambda=tf_lambda, **td3_ddpg_kwargs)
    elif agent_name == 'SAC':
        model = SAC('MlpPolicy', env, verbose=2, device=run_device, buffer_size=buffer_size, tf_lambda=tf_lambda,
                    **sac_kwargs)
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
