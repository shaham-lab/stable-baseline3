from typing import Any, Optional, TypeVar, Union

import torch as th

from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.td3.td3_sl import TD3SL

SelfDDPGSL = TypeVar("SelfDDPGSL", bound="DDPGSL")


class DDPGSL(TD3SL):
    """
    Deep Deterministic Policy Gradient (DDPG) with a short term and a long term replay buffer
    (see ``ShortLongTermReplayBuffer``)

    Deterministic Policy Gradient: https://proceedings.mlr.press/v32/silver14.pdf
    DDPG Paper: https://arxiv.org/abs/1509.02971
    Introduction to DDPG: https://spinningup.openai.com/en/latest/algorithms/ddpg.html

    Note: we treat DDPG as a special case of its successor TD3.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the short term replay buffer
    :param long_term_buffer_size: size of the long term replay buffer
    :param sampling_fraction: fraction of each minibatch taken from the short term replay buffer
        (the rest is taken from the long term replay buffer, once it has enough samples)
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_kwargs: Additional keyword arguments to pass to the replay buffer on creation.
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`ddpg_policies`
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 4000,  # short buffer 4K
        long_term_buffer_size: int = 1_000_000,  # 1e6
        sampling_fraction: float = 0.5,
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 4,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        tf_lambda: float = 0.0  # SUFT CHANGE
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            long_term_buffer_size=long_term_buffer_size,
            sampling_fraction=sampling_fraction,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            # Remove all tricks from TD3 to obtain DDPG:
            # we still need to specify target_policy_noise > 0 to avoid errors
            policy_delay=1,
            target_noise_clip=0.0,
            target_policy_noise=0.1,
            _init_setup_model=False,
            tf_lambda=tf_lambda,  # SUFT CHANGE
        )

        # Use only one critic
        if "n_critics" not in self.policy_kwargs:
            self.policy_kwargs["n_critics"] = 1

        if _init_setup_model:
            self._setup_model()

    def learn(
        self: SelfDDPGSL,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DDPGSL",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDDPGSL:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
