import os
import sys
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
    run_device = sys.argv[1]
    print("device", run_device)
    # get CF lambda from command line
    tf_lambda = float(sys.argv[2])
    print("tf_lambda", tf_lambda)
    # get theory from command line
    theory = sys.argv[3]
    print("theory", theory)
    # get the agent
    agent_name = sys.argv[4]
    print("agent_name", agent_name)
    # get the timestep from command line
    timesteps = int(sys.argv[5])
    print("timesteps", timesteps)
    # get the run number from command line
    run_number = sys.argv[6]
    print("run_number", run_number)
    # get the random seed
    random_seed = int(sys.argv[7])
    print("random_seed", random_seed)
    # get the environment from command line
    gym_env_name = sys.argv[8]
    print("gym_env_name", gym_env_name)

    # get the model name from command line
    model_name = gym_env_name + "_" + agent_name + "_atari_tf_lambda_" + str(tf_lambda) + "_timesteps_" + str(timesteps) + "_run_" + str(
        run_number) + "_seed_" + str(random_seed)
    model_name = model_name.replace("ALE/", "")
    print("model_name", model_name)

    # Full directory path
    full_log_dir = os.path.join("/home/dsi/fiskustal/talfiskus/stable-baselines3/", theory, agent_name, gym_env_name.replace("ALE/", ""))
    if not os.path.exists(full_log_dir):
        os.makedirs(full_log_dir, exist_ok=True)
    print("full_log_dir", full_log_dir)

    # v5: disable the built-in frame skip (AtariWrapper already skips 4 frames)
    # and let ALE alone handle sticky actions
    atari_env_kwargs = {"frameskip": 1, "repeat_action_probability": 0.25}
    env = make_atari_env(gym_env_name, seed=random_seed, env_kwargs=atari_env_kwargs)
    # Stack 4 frames
    env = VecFrameStack(env, n_stack=4)
    # Eval env
    eval_env = make_atari_env(gym_env_name, seed=random_seed, env_kwargs=atari_env_kwargs)
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_callback = EvalCallback(eval_env, eval_freq=10000, verbose=1, n_eval_episodes=100)
    # select agent by name
    if agent_name == 'DQN':
        buffer_size = int(sys.argv[9])
        print("buffer_size", buffer_size)
        model = DQN('CnnPolicy', env, verbose=2, device=run_device, tf_lambda=tf_lambda, buffer_size=buffer_size)
    elif agent_name == 'DDQN':
        buffer_size = int(sys.argv[9])
        print("buffer_size", buffer_size)
        model = DDQN('CnnPolicy', env, verbose=2, device=run_device, tf_lambda=tf_lambda, buffer_size=buffer_size)
    elif agent_name == 'DQN1':
        buffer_size = int(sys.argv[9])
        print("buffer_size", buffer_size)
        model = DQN1('CnnPolicy', env, verbose=2, device=run_device, tf_lambda=tf_lambda, buffer_size=buffer_size)

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

    # # Enjoy trained agent
    # video_length = 2000
    # vec_env = model.get_env()
    # # Record the video starting at the first step
    # vec_env = VecVideoRecorder(vec_env, f"{full_log_dir}/video",
    #                            record_video_trigger=lambda x: x == 0, video_length=video_length,
    #                            name_prefix=f"{model_name}")

    # obs = vec_env.reset()
    # for i in range(video_length):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, rewards, dones, info = vec_env.step(action)
    # # Save the video
    # vec_env.close()
