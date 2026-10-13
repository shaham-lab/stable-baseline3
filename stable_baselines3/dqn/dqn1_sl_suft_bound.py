import warnings
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ShortLongTermReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, ReplayBufferSamples, Schedule
from stable_baselines3.common.utils import LinearSchedule, get_parameters_by_name, polyak_update
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork

SelfDQN1SLSB = TypeVar("SelfDQN1SLSB", bound="DQN1SLSB")


class DQN1SLSB(OffPolicyAlgorithm):
    """
    Deep Q-Network (DQN- 2013) with a short term and a long term replay buffer
    (see ``ShortLongTermReplayBuffer``)

    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default hyperparameters are taken from the Nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the short term replay buffer
    :param long_term_buffer_size: size of the long term replay buffer
    :param sampling_fraction: fraction of each minibatch taken from the short term replay buffer
        (the rest is taken from the long term replay buffer, once it has enough samples)
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_kwargs: Additional keyword arguments to pass to the replay buffer on creation.
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
        After every gradient update, their SUFT term is printed (0 to disable)
    :param track_behavior_distance: Whether to print the distance between the current weights and the weights
        of the behavior policy of each sample of the minibatch (the gradient updates, actions taken
        and timesteps differences are always printed). It keeps a full copy of the weights
        for every optimizer step still referenced by the replay buffer, which takes a lot of memory.
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
    # Statistics comparing the current Q-network with the behavior policy of the samples, see `_behavior_stats()`
    behavior_stats_names: ClassVar[tuple[str, ...]] = (
        "behavior distance mean",
        "behavior distance^2 mean",
        "gradient updates difference max",
        "actions taken difference max",
        "timesteps difference max",
    )

    def __init__(
        self,
        policy: Union[str, type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 4000,  # short buffer 4K
        long_term_buffer_size: int = 1_000_000,  # 1e6
        sampling_fraction: float = 0.5,
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
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
        track_behavior_distance: bool = False,
    ) -> None:
        # Short term and long term replay buffer, `buffer_size` is the size of the short term one
        replay_buffer_kwargs = {
            **(replay_buffer_kwargs or {}),
            "long_term_buffer_size": long_term_buffer_size,
            "sampling_fraction": sampling_fraction,
        }
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
            replay_buffer_class=ShortLongTermReplayBuffer,
            replay_buffer_kwargs=replay_buffer_kwargs,
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
        print("short term buffer size", buffer_size)
        print("long term buffer size", long_term_buffer_size)
        print("sampling fraction", sampling_fraction)
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
        # Weights of the behavior policies: the ones of the Q-network after each optimizer step
        # that experiences still in the replay buffer were collected with (kept on the CPU)
        self.track_behavior_distance = track_behavior_distance
        self.behavior_q_net_params: dict[int, th.Tensor] = {}

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
            # No gradient update yet: keep the first experiences (all still in the short term buffer) in the cold buffer
            n_cold = min(self.cold_buffer_size, self.replay_buffer.size())  # type: ignore[union-attr]
            self.cold_buffer = self.replay_buffer._get_samples(np.arange(n_cold), env=self._vec_normalize_env)  # type: ignore[union-attr]
            self.cold_q_net_params = self._get_q_net_params()
        if self.track_behavior_distance and self._n_updates == 0 and 0 not in self.behavior_q_net_params:
            # Weights the experiences collected before any gradient update come from
            self.behavior_q_net_params[0] = self._get_q_net_params().cpu()

        losses = []
        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            short_replay_data, long_replay_data = self.replay_buffer.sample_short_and_long(batch_size, env=self._vec_normalize_env)
            short_replay_loss_sum, short_suft_mean = self.calculate_loss(short_replay_data, True)
            long_replay_loss_sum, long_suft_mean = self.calculate_loss(long_replay_data, False)
            loss = (short_replay_loss_sum + long_replay_loss_sum)/batch_size
            losses.append(loss.item())
            # Compare with the behavior policies, with the same weights as the SUFT terms (before the update)
            short_behavior_stats = self._behavior_stats(short_replay_data, self._n_updates + gradient_step + 1)
            long_behavior_stats = self._behavior_stats(long_replay_data, self._n_updates + gradient_step + 1)

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            if self.cold_buffer is not None:
                self._print_cold_buffer_suft(
                    self._n_updates + gradient_step + 1,
                    short_suft_mean,
                    long_suft_mean,
                    short_behavior_stats,
                    long_behavior_stats,
                )

        # Increase update counter
        self._n_updates += gradient_steps
        self._update_behavior_q_net_params()

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def calculate_loss(
        self, replay_data: Optional[ReplayBufferSamples], is_short: bool
    ) -> tuple[Union[th.Tensor, float], float]:
        """
        Compute the loss summed over the samples (the caller divides by the batch size).

        :param replay_data: Samples from one of the two buffers (``None`` if nothing was sampled from it)
        :param is_short: Whether the samples come from the short term buffer,
            the SUFT term is only added for those
        :return: Sum over the samples of the TD loss (plus the SUFT term scaled by ``tf_lambda``
            for the short term samples), ``0.0`` if nothing was sampled,
            and the mean SUFT term of the samples (not scaled by ``tf_lambda``, only for printing),
            ``nan`` if nothing was sampled
        """
        if replay_data is None:
            return 0.0, float("nan")
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
        loss = F.mse_loss(current_q_values, target_q_values, reduction='sum')
        # SUFT CHANGE
        old_q_values = th.gather(replay_data.old_values, dim=1, index=replay_data.actions.long())
        # SUFT term summed over the samples (not scaled by tf_lambda)
        suft_sum = F.mse_loss(current_q_values, old_q_values, reduction='sum')
        if is_short and self.tf_lambda > 0:
            suft_ope_term = suft_sum * self.tf_lambda
            loss = loss + suft_ope_term
        # SUFT CHANGE until here
        return loss, suft_sum.item() / len(current_q_values)

    def _get_q_net_params(self) -> th.Tensor:
        """
        Flattened copy of all the weights of the Q-network.
        """
        return th.cat([param.detach().flatten() for param in self.q_net.parameters()])

    def _behavior_stats(self, replay_data: Optional[ReplayBufferSamples], n_updates: int) -> dict[str, float]:
        """
        Compare the current Q-network with the behavior policy of each sample
        (the Q-network when the sample was collected).

        :param replay_data: Samples of the current minibatch (``None`` if nothing was sampled)
        :param n_updates: Number of gradient updates done so far, including the current one (the printed one)
        :return: Mean over the samples of the L2 distance between the weights and of the squared distance,
            and max over the samples (the one of the oldest sample) of the number of gradient updates,
            of actions taken and of timesteps since the sample was collected (``nan`` when not available)
        """
        stats = dict.fromkeys(self.behavior_stats_names, float("nan"))
        if replay_data is None or replay_data.optimizer_steps is None:
            return stats
        stats["gradient updates difference max"] = n_updates - replay_data.optimizer_steps.min().item()
        actions_difference = self.num_timesteps - replay_data.timesteps.min().item()
        stats["actions taken difference max"] = actions_difference
        stats["timesteps difference max"] = actions_difference // self.n_envs
        q_net_params = self._get_q_net_params()
        steps, counts = np.unique(replay_data.optimizer_steps.cpu().numpy(), return_counts=True)
        distance_sum, distance_sq_sum, n_samples = 0.0, 0.0, 0
        for step, count in zip(steps, counts):
            behavior_params = self.behavior_q_net_params.get(int(step))
            if behavior_params is None:
                continue
            distance = th.linalg.vector_norm(q_net_params - behavior_params.to(q_net_params.device)).item()
            distance_sum += distance * count
            distance_sq_sum += distance**2 * count
            n_samples += count
        if n_samples > 0:
            stats["behavior distance mean"] = distance_sum / n_samples
            stats["behavior distance^2 mean"] = distance_sq_sum / n_samples
        return stats

    def _store_transition(self, replay_buffer, *args, **kwargs) -> None:
        # Tag the transition with the number of actions taken so far
        replay_buffer.current_timestep = self.num_timesteps
        super()._store_transition(replay_buffer, *args, **kwargs)

    def _update_behavior_q_net_params(self) -> None:
        """
        Keep the weights the next experiences are collected with,
        and drop the ones that no experience of the replay buffer was collected with anymore.
        """
        self.replay_buffer.current_optimizer_step = self._n_updates  # type: ignore[union-attr]
        if not self.track_behavior_distance:
            return
        self.behavior_q_net_params[self._n_updates] = self._get_q_net_params().cpu()
        oldest_step = self.replay_buffer.oldest_optimizer_step()  # type: ignore[union-attr]
        for step in [step for step in self.behavior_q_net_params if step < oldest_step]:
            del self.behavior_q_net_params[step]

    def _print_cold_buffer_suft(
        self,
        n_updates: int,
        short_suft_mean: float,
        long_suft_mean: float,
        short_behavior_stats: dict[str, float],
        long_behavior_stats: dict[str, float],
    ) -> None:
        """
        Print the number of actions taken and of gradient updates so far, the learning rate,
        the mean SUFT term of the cold buffer (the squared difference between the current Q-value and the one stored
        when the experience was collected, not scaled by ``tf_lambda``), the ones of the current minibatch,
        and the distance between the current weights and the ones before any gradient update.

        :param n_updates: Number of gradient updates done so far
        :param short_suft_mean: SUFT term of the short term samples of this gradient update (not scaled by ``tf_lambda``)
        :param long_suft_mean: SUFT term of the long term samples of this gradient update,
            it is not part of the loss (``nan`` if there are none)
        :param short_behavior_stats: Comparison between the Q-network before this gradient update
            and the behavior policy of each short term sample, see ``_behavior_stats()``
        :param long_behavior_stats: Same for the long term samples (``nan`` if there are none)
        """
        with th.no_grad():
            current_q_values = self.q_net(self.cold_buffer.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=self.cold_buffer.actions.long())
            old_q_values = th.gather(self.cold_buffer.old_values, dim=1, index=self.cold_buffer.actions.long())
            suft_terms = F.mse_loss(current_q_values, old_q_values)
            # L2 distance between the current weights and the ones before any gradient update
            distance = th.linalg.vector_norm(self._get_q_net_params() - self.cold_q_net_params)
        learning_rate = self.policy.optimizer.param_groups[0]["lr"]
        print(f"actions taken {self.num_timesteps} | timestep {self.num_timesteps // self.n_envs} | "
              f"gradient updates {n_updates} | learning rate {learning_rate} | "
              f"cold buffer SUFT mean {suft_terms:.2e} | current short term SUFT mean {short_suft_mean:.2e} | "
              f"current long term SUFT mean {long_suft_mean:.2e} | "
              f"distance {distance:.2e} | distance^2 {distance ** 2:.2e} | "
              + " | ".join(f"current short term {name} {value if isinstance(value, int) else format(value, '.2e')}" for name, value in short_behavior_stats.items())
              + " | "
              + " | ".join(f"current long term {name} {value if isinstance(value, int) else format(value, '.2e')}" for name, value in long_behavior_stats.items()))

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
        self: SelfDQN1SLSB,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DQN1SLSB",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDQN1SLSB:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "q_net", "cold_buffer", "cold_q_net_params", "behavior_q_net_params"]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
