import warnings
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, ReplayBufferSamples, Schedule
from stable_baselines3.common.utils import LinearSchedule, get_parameters_by_name, polyak_update
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork

SelfDQN1 = TypeVar("SelfDQN1", bound="DQN1")


class DQN1(OffPolicyAlgorithm):
    """
    Deep Q-Network (DQN- 2013)

    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default hyperparameters are taken from the Nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param n_steps: When n_step > 1, uses n-step return (with the NStepReplayBuffer) when updating the Q-value network.
        Note: it is only used when ``replay_buffer_class`` is ``None``, and is not supported for Dict
        observation spaces yet. When passing a custom ``replay_buffer_class``, configure it directly
        with ``replay_buffer_kwargs={"n_steps": ..., "gamma": ...}`` instead.
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`dqn_policies`
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param cold_buffer_size: Number of experiences, collected before any gradient update, kept in the cold buffer.
        After every gradient update, the SUFT term of each one of them is printed (0 to disable)
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    # Linear schedule will be defined in `_setup_model()`
    exploration_schedule: Schedule
    q_net: QNetwork
    policy: DQNPolicy

    def __init__(
        self,
        policy: Union[str, type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        tf_lambda: float = 0.0,  # SUFT CHANGE
        cold_buffer_size: int = 100,
    ) -> None:
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=True,
            is_dqn=True,
        )
        print("Loss", "MSE")
        print("buffer size", buffer_size)
        print("TF lambda", tf_lambda)
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0
        self.tf_lambda = tf_lambda  # SUFT CHANGE
        # Cold buffer: frozen copy of the first experiences, collected before any gradient update
        self.cold_buffer_size = cold_buffer_size
        self.cold_buffer: Optional[ReplayBufferSamples] = None
        # Weights of the Q-network before any gradient update (the ones the cold buffer was collected with)
        self.cold_q_net_params: Optional[th.Tensor] = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Copy running stats, see GH issue #996
        self.batch_norm_stats = get_parameters_by_name(self.q_net, ["running_"])
        self.exploration_schedule = LinearSchedule(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net

    def _on_step(self) -> None:
        """
        Update the exploration rate
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        if self.cold_buffer is None and self.cold_buffer_size > 0 and self._n_updates == 0:
            # No gradient update yet: keep the first experiences in the cold buffer
            n_cold = min(self.cold_buffer_size, self.replay_buffer.size())  # type: ignore[union-attr]
            self.cold_buffer = self.replay_buffer._get_samples(np.arange(n_cold), env=self._vec_normalize_env)  # type: ignore[union-attr]
            self.cold_q_net_params = th.cat([param.detach().flatten().clone() for param in self.q_net.parameters()])

        losses = []
        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            with th.no_grad():
                # Compute the next Q-values using the current network
                next_q_values = self.q_net(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())
            # Compute MSE loss
            loss = F.mse_loss(current_q_values, target_q_values)
            # SUFT CHANGE
            old_q_values = th.gather(replay_data.old_values, dim=1, index=replay_data.actions.long())
            # SUFT term of the current minibatch (not scaled by tf_lambda)
            current_suft_mean = F.mse_loss(current_q_values, old_q_values)
            if self.tf_lambda > 0:
                suft_ope_term = current_suft_mean * self.tf_lambda
                loss = loss + suft_ope_term
            # SUFT CHANGE until here
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            if self.cold_buffer is not None:
                self._print_cold_buffer_suft(self._n_updates + gradient_step + 1, current_suft_mean.item())

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def _print_cold_buffer_suft(self, n_updates: int, current_suft_mean: float) -> None:
        """
        Print the number of gradient updates so far, the learning rate and the SUFT term of each experience
        of the cold buffer: the squared difference between the current Q-value and the one stored
        when the experience was collected (not scaled by ``tf_lambda``).

        :param n_updates: Number of gradient updates done so far
        :param current_suft_mean: SUFT term of the minibatch of this gradient update (not scaled by ``tf_lambda``)
        """
        with th.no_grad():
            current_q_values = self.q_net(self.cold_buffer.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=self.cold_buffer.actions.long())
            old_q_values = th.gather(self.cold_buffer.old_values, dim=1, index=self.cold_buffer.actions.long())
            suft_terms = F.mse_loss(current_q_values, old_q_values)
            # L2 distance between the current weights and the ones before any gradient update
            q_net_params = th.cat([param.detach().flatten() for param in self.q_net.parameters()])
            distance = th.linalg.vector_norm(q_net_params - self.cold_q_net_params)
        learning_rate = self.policy.optimizer.param_groups[0]["lr"]
        print(f"actions taken {self.num_timesteps} | gradient updates {n_updates} | learning rate {learning_rate} | "
              f"cold buffer SUFT mean {suft_terms:.6e} | current SUFT mean {current_suft_mean:.6e} | "
              f"distance {distance:.6e} | distance^2 {distance ** 2:.6e}")

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[next(iter(observation.keys()))].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state

    def learn(
        self: SelfDQN1,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDQN1:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "q_net", "cold_buffer", "cold_q_net_params"]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
