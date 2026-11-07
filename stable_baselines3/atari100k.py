import os
import sys

from stable_baselines3.dqn.ddqn_sl import DDQNSL
from stable_baselines3.dqn.dqn1_sl import DQN1SL
from stable_baselines3.dqn.dqn_sl import DQNSL

sys.path.append("/home/dsi/fiskustal/talfiskus/stable-baselines3-new/stable-baseline3/")

from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.dqn.ddqn import DDQN
from stable_baselines3.dqn.dqn1 import DQN1
from stable_baselines3.dqn.dqn1_sl_suft_bound import DQN1SLSB
from stable_baselines3.dqn.dqn1_suft_bound import DQN1SB
from stable_baselines3.dqn.dqn_n_step import DQNNStep
from stable_baselines3.dqn.dqnsl_n_step import DQNSLNStep
from stable_baselines3.dqn.dqnper import DQNPER
from stable_baselines3.dqn.dqnsl_per_long import DQNSLPERLong
from stable_baselines3.dqn.dqnsl_per_both import DQNSLPERBoth
from stable_baselines3.dqn.dqncer import DQNCER
from stable_baselines3.dqn.dqnsl_cer import DQNSLCER
from stable_baselines3.dqn.dqn_dueling import DQNDueling
from stable_baselines3.dqn.dqnsl_dueling import DQNSLDueling
from stable_baselines3.dqn.ddqn_n_step_per_dueling import DDQNNStepPERDueling
from stable_baselines3.dqn.ddqnsl_n_step_per_dueling import DDQNSLNStepPERDueling
from stable_baselines3.dqn.dqn_der import DQNDER
from stable_baselines3.dqn.dqnsl_der import DQNSLDER
from stable_baselines3.dqn.dqn_ere import DQNERE
from stable_baselines3.dqn.dqnsl_ere import DQNSLERE
from stable_baselines3.common.callbacks import EvalCallback

import time
from datetime import timedelta

if __name__ == "__main__":
    run_device = sys.argv[1]
    print("device", run_device)
    # get CF lambda from command line
    tf_lambda = float(sys.argv[2])
    print("tf_lambda", tf_lambda)
    # get the agent
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
    buffer_size = int(sys.argv[7])
    print("buffer_size", buffer_size)
    # get the model name from command line
    model_name = gym_env_name + "_" + agent_name + "_atari_tf_lambda_" + str(tf_lambda) + "_timesteps_" + str(timesteps) + "_run_" + str(
        run_number)
    model_name = model_name.replace("ALE/", "")
    print("model_name", model_name)

    # Full directory path
    full_log_dir = os.path.join("/home/dsi/fiskustal/talfiskus/stable-baselines3/", agent_name, gym_env_name.replace("ALE/", ""))
    if not os.path.exists(full_log_dir):
        os.makedirs(full_log_dir, exist_ok=True)
    print("full_log_dir", full_log_dir)

    # v5: disable the built-in frame skip (AtariWrapper already skips 4 frames)
    # and let ALE alone handle sticky actions
    atari_env_kwargs = {"frameskip": 1, "repeat_action_probability": 0.25}
    env = make_atari_env(gym_env_name, env_kwargs=atari_env_kwargs)
    # Stack 4 frames
    env = VecFrameStack(env, n_stack=4)
    # Eval env
    eval_env = make_atari_env(gym_env_name, env_kwargs=atari_env_kwargs)
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_callback = EvalCallback(eval_env, eval_freq=100000, verbose=1, n_eval_episodes=100, deterministic=False)
    # select agent by name
    if agent_name == 'DQN':
        model = DQN('CnnPolicy', env, verbose=2, device=run_device, tf_lambda=tf_lambda, buffer_size=buffer_size)
    elif agent_name == 'DDQN':
        model = DDQN('CnnPolicy', env, verbose=2, device=run_device, tf_lambda=tf_lambda, buffer_size=buffer_size)
    elif agent_name == 'DQN1':
        model = DQN1('CnnPolicy', env, verbose=2, device=run_device, tf_lambda=tf_lambda, buffer_size=buffer_size)
    elif agent_name == 'DQNSL':
        short_buffer_size = int(sys.argv[8])
        print("short_buffer_size", short_buffer_size)
        sampling_fraction = float(sys.argv[9])
        print("sampling_fraction", sampling_fraction)
        model = DQNSL('CnnPolicy', env, verbose=2, device=run_device, tf_lambda=tf_lambda, buffer_size=short_buffer_size,
                      long_term_buffer_size=buffer_size, sampling_fraction=sampling_fraction)
    elif agent_name == 'DQN1SL':
        short_buffer_size = int(sys.argv[8])
        print("short_buffer_size", short_buffer_size)
        sampling_fraction = float(sys.argv[9])
        print("sampling_fraction", sampling_fraction)
        model = DQN1SL('CnnPolicy', env, verbose=2, device=run_device, tf_lambda=tf_lambda, buffer_size=short_buffer_size,
                      long_term_buffer_size=buffer_size, sampling_fraction=sampling_fraction)
    elif agent_name == 'DQN1SB':
        # optional, keeps a copy of the weights per gradient update (too much memory with a big buffer)
        track_behavior_distance = sys.argv[8].lower() in ("true", "1") if len(sys.argv) > 8 else False
        print("track_behavior_distance", track_behavior_distance)
        model = DQN1SB('CnnPolicy', env, verbose=2, device=run_device, tf_lambda=tf_lambda, buffer_size=buffer_size,
                       track_behavior_distance=track_behavior_distance)
    elif agent_name == 'DQN1SLSB':
        short_buffer_size = int(sys.argv[8])
        print("short_buffer_size", short_buffer_size)
        sampling_fraction = float(sys.argv[9])
        print("sampling_fraction", sampling_fraction)
        # optional, keeps a copy of the weights per gradient update (too much memory with a big buffer)
        track_behavior_distance = sys.argv[10].lower() in ("true", "1") if len(sys.argv) > 10 else False
        print("track_behavior_distance", track_behavior_distance)
        model = DQN1SLSB('CnnPolicy', env, verbose=2, device=run_device, tf_lambda=tf_lambda, buffer_size=short_buffer_size,
                         long_term_buffer_size=buffer_size, sampling_fraction=sampling_fraction,
                         track_behavior_distance=track_behavior_distance)
    elif agent_name == 'DDQNSL':
        short_buffer_size = int(sys.argv[8])
        print("short_buffer_size", short_buffer_size)
        sampling_fraction = float(sys.argv[9])
        print("sampling_fraction", sampling_fraction)
        model = DDQNSL('CnnPolicy', env, verbose=2, device=run_device, tf_lambda=tf_lambda, buffer_size=short_buffer_size,
                      long_term_buffer_size=buffer_size, sampling_fraction=sampling_fraction)
    elif agent_name == 'DQNNStep':
        # optional n-step length, Rainbow default is 3
        n_steps = int(sys.argv[8]) if len(sys.argv) > 8 else 3
        print("n_steps", n_steps)
        model = DQNNStep('CnnPolicy', env, verbose=2, device=run_device, tf_lambda=tf_lambda, buffer_size=buffer_size,
                         n_steps=n_steps)
    elif agent_name == 'DQNSLNStep':
        short_buffer_size = int(sys.argv[8])
        print("short_buffer_size", short_buffer_size)
        sampling_fraction = float(sys.argv[9])
        print("sampling_fraction", sampling_fraction)
        # optional n-step length, Rainbow default is 3
        n_steps = int(sys.argv[10]) if len(sys.argv) > 10 else 3
        print("n_steps", n_steps)
        model = DQNSLNStep('CnnPolicy', env, verbose=2, device=run_device, tf_lambda=tf_lambda, buffer_size=short_buffer_size,
                           long_term_buffer_size=buffer_size, sampling_fraction=sampling_fraction, n_steps=n_steps)
    elif agent_name == 'DQNPER':
        model = DQNPER('CnnPolicy', env, verbose=2, device=run_device, tf_lambda=tf_lambda, buffer_size=buffer_size)
    elif agent_name == 'DQNSLPERLong':
        short_buffer_size = int(sys.argv[8])
        print("short_buffer_size", short_buffer_size)
        sampling_fraction = float(sys.argv[9])
        print("sampling_fraction", sampling_fraction)
        model = DQNSLPERLong('CnnPolicy', env, verbose=2, device=run_device, tf_lambda=tf_lambda, buffer_size=short_buffer_size,
                             long_term_buffer_size=buffer_size, sampling_fraction=sampling_fraction)
    elif agent_name == 'DQNSLPERBoth':
        short_buffer_size = int(sys.argv[8])
        print("short_buffer_size", short_buffer_size)
        sampling_fraction = float(sys.argv[9])
        print("sampling_fraction", sampling_fraction)
        model = DQNSLPERBoth('CnnPolicy', env, verbose=2, device=run_device, tf_lambda=tf_lambda, buffer_size=short_buffer_size,
                             long_term_buffer_size=buffer_size, sampling_fraction=sampling_fraction)
    elif agent_name == 'DQNCER':
        # optional number of latest transitions in every batch, default train_freq (1 = original CER)
        cer_n_latest = int(sys.argv[8]) if len(sys.argv) > 8 else None
        print("cer_n_latest", cer_n_latest if cer_n_latest is not None else "train_freq")
        model = DQNCER('CnnPolicy', env, verbose=2, device=run_device, tf_lambda=tf_lambda, buffer_size=buffer_size,
                       cer_n_latest=cer_n_latest)
    elif agent_name == 'DQNSLCER':
        short_buffer_size = int(sys.argv[8])
        print("short_buffer_size", short_buffer_size)
        sampling_fraction = float(sys.argv[9])
        print("sampling_fraction", sampling_fraction)
        # optional number of latest transitions in every batch, default train_freq (1 = original CER)
        cer_n_latest = int(sys.argv[10]) if len(sys.argv) > 10 else None
        print("cer_n_latest", cer_n_latest if cer_n_latest is not None else "train_freq")
        model = DQNSLCER('CnnPolicy', env, verbose=2, device=run_device, tf_lambda=tf_lambda, buffer_size=short_buffer_size,
                         long_term_buffer_size=buffer_size, sampling_fraction=sampling_fraction, cer_n_latest=cer_n_latest)
    elif agent_name == 'DQNDueling':
        # Dueling network architecture (Wang et al., 2016) on the DQN with target network, SUFT term with tf_lambda > 0
        model = DQNDueling('CnnPolicy', env, verbose=2, device=run_device, tf_lambda=tf_lambda, buffer_size=buffer_size)
    elif agent_name == 'DQNSLDueling':
        # Dueling network architecture with the short and long term buffers (SUFT term on the short term samples)
        short_buffer_size = int(sys.argv[8])
        print("short_buffer_size", short_buffer_size)
        sampling_fraction = float(sys.argv[9])
        print("sampling_fraction", sampling_fraction)
        model = DQNSLDueling('CnnPolicy', env, verbose=2, device=run_device, tf_lambda=tf_lambda, buffer_size=short_buffer_size,
                             long_term_buffer_size=buffer_size, sampling_fraction=sampling_fraction)
    elif agent_name == 'DDQNNStepPERDueling':
        # Double DQN + n-step + PER (alpha 0.5) + dueling, on the DQN with target network, SUFT term with tf_lambda > 0
        # optional n-step length, Rainbow default is 3
        n_steps = int(sys.argv[8]) if len(sys.argv) > 8 else 3
        print("n_steps", n_steps)
        model = DDQNNStepPERDueling('CnnPolicy', env, verbose=2, device=run_device, tf_lambda=tf_lambda, buffer_size=buffer_size,
                                    n_steps=n_steps)
    elif agent_name == 'DDQNSLNStepPERDueling':
        # Short-long SUFT version (SUFT term on the short term samples)
        short_buffer_size = int(sys.argv[8])
        print("short_buffer_size", short_buffer_size)
        sampling_fraction = float(sys.argv[9])
        print("sampling_fraction", sampling_fraction)
        # optional n-step length, Rainbow default is 3
        n_steps = int(sys.argv[10]) if len(sys.argv) > 10 else 3
        print("n_steps", n_steps)
        # optional, whether the short term buffer is prioritized too (default true, false: sampled uniformly)
        prioritize_short_term = sys.argv[11].lower() in ("true", "1") if len(sys.argv) > 11 else True
        print("prioritize_short_term", prioritize_short_term)
        model = DDQNSLNStepPERDueling('CnnPolicy', env, verbose=2, device=run_device, tf_lambda=tf_lambda,
                                      buffer_size=short_buffer_size, long_term_buffer_size=buffer_size,
                                      sampling_fraction=sampling_fraction, n_steps=n_steps,
                                      prioritize_short_term=prioritize_short_term)
    elif agent_name == 'DQNDER':
        # Double experience replay (Han et al., 2021) on the DQN with target network, buffer_size = H1 + H2 budget,
        # SUFT term (both memories) with tf_lambda > 0
        # optional fraction of each batch taken from H2 (lambda of the paper), default 0.1
        der_lambda = float(sys.argv[8]) if len(sys.argv) > 8 else 0.1
        print("der_lambda", der_lambda)
        model = DQNDER('CnnPolicy', env, verbose=2, device=run_device, tf_lambda=tf_lambda, buffer_size=buffer_size,
                       der_lambda=der_lambda)
    elif agent_name == 'DQNSLDER':
        # Short-long SUFT: uniform short term buffer + long term DER (budget buffer_size = 1M - short buffer),
        # SUFT term on the short term samples
        short_buffer_size = int(sys.argv[8])
        print("short_buffer_size", short_buffer_size)
        sampling_fraction = float(sys.argv[9])
        print("sampling_fraction", sampling_fraction)
        # optional fraction of the long term part of each batch taken from H2, default 0.1
        der_lambda = float(sys.argv[10]) if len(sys.argv) > 10 else 0.1
        print("der_lambda", der_lambda)
        model = DQNSLDER('CnnPolicy', env, verbose=2, device=run_device, tf_lambda=tf_lambda, buffer_size=short_buffer_size,
                         long_term_buffer_size=buffer_size, sampling_fraction=sampling_fraction, der_lambda=der_lambda)
    elif agent_name == 'DQNERE':
        # Emphasizing recent experience (Wang et al., 2020) on the DQN with target network, SUFT term with tf_lambda > 0
        # optional: also do the usual DQN updates (replay ratio 0.25) besides the ERE updates after every episode
        ere_continuous_updates = sys.argv[8].lower() in ("true", "1") if len(sys.argv) > 8 else False
        print("ere_continuous_updates", ere_continuous_updates)
        model = DQNERE('CnnPolicy', env, verbose=2, device=run_device, tf_lambda=tf_lambda, buffer_size=buffer_size,
                       ere_continuous_updates=ere_continuous_updates)
    elif agent_name == 'DQNSLERE':
        # Short-long SUFT, uniform short term buffer + ERE on the long term buffer (SUFT term on the short term samples)
        short_buffer_size = int(sys.argv[8])
        print("short_buffer_size", short_buffer_size)
        sampling_fraction = float(sys.argv[9])
        print("sampling_fraction", sampling_fraction)
        # optional: also do the usual DQN updates (replay ratio 0.25) besides the ERE updates after every episode
        ere_continuous_updates = sys.argv[10].lower() in ("true", "1") if len(sys.argv) > 10 else False
        print("ere_continuous_updates", ere_continuous_updates)
        model = DQNSLERE('CnnPolicy', env, verbose=2, device=run_device, tf_lambda=tf_lambda, buffer_size=short_buffer_size,
                         long_term_buffer_size=buffer_size, sampling_fraction=sampling_fraction,
                         ere_continuous_updates=ere_continuous_updates)

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
