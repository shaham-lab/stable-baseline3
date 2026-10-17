import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    PrioritizedReplayBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    observation_space: spaces.Space
    obs_shape: tuple[int, ...]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)  # type: ignore[assignment]

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)

    @staticmethod
    def _normalize_obs(
        obs: Union[np.ndarray, dict[str, np.ndarray]],
        env: Optional[VecNormalize] = None,
    ) -> Union[np.ndarray, dict[str, np.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray, env: Optional[VecNormalize] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray
    old_values_to_store: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        is_dqn: bool = False,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Flag forwarded by the agent:
        # DQN / DDQN / DQN1 (Q-values of all actions), otherwise SAC (one value per critic)
        self.is_dqn = is_dqn
        if self.is_dqn:
            # DQN
            self.old_values = np.zeros((self.buffer_size, self.n_envs, self.action_space.n), dtype=np.float32)
        else:
            # SAC
            self.old_values1 = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.float32)
            self.old_values2 = np.zeros((self.buffer_size, self.n_envs, 1), dtype=np.float32)
        self.optimizer_steps = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes
                + self.actions.nbytes
                + self.rewards.nbytes
                + self.dones.nbytes
                + self.optimizer_steps.nbytes
            )
            if self.is_dqn:
                # DQN
                total_memory_usage += self.old_values.nbytes
            else:
                # SAC
                total_memory_usage += self.old_values1.nbytes + self.old_values2.nbytes

            if not optimize_memory_usage:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        old_values_to_store,
        infos: list[dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)
        if self.is_dqn:
            # DQN
            self.old_values[self.pos] = old_values_to_store.detach().cpu().numpy()
        else:
            # SAC
            first_old = old_values_to_store[0].detach().cpu().numpy()
            # last critic (the second one for SAC)
            second_old = old_values_to_store[-1].detach().cpu().numpy()
            self.old_values1[self.pos] = np.array(first_old)
            self.old_values2[self.pos] = np.array(second_old)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        if self.is_dqn:
            # DQN
            old_values = {
                "old_values": self.to_torch(self.old_values[batch_inds, env_indices]),
            }
        else:
            # SAC
            old_values = {
                "old_values1": self.to_torch(self.old_values1[batch_inds, env_indices]),
                "old_values2": self.to_torch(self.old_values2[batch_inds, env_indices]),
            }
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)), **old_values)

    @staticmethod
    def _maybe_cast_dtype(dtype: Optional[np.typing.DTypeLike]) -> Optional[np.typing.DTypeLike]:
        """
        Cast `np.float64` action datatype to `np.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype


class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=self.observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=self.action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super().reset()

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()  # type: ignore[assignment]

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            # Cast to float32 (backward compatible), this would lead to RuntimeError for MultiBinary space
            self.actions[batch_inds].astype(np.float32, copy=False),
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))


class DictReplayBuffer(ReplayBuffer):
    """
    Dict Replay buffer used in off-policy algorithms like SAC/TD3.
    Extends the ReplayBuffer to use dictionary observations

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        Disabled for now (see https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702)
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observation_space: spaces.Dict
    obs_shape: dict[str, tuple[int, ...]]  # type: ignore[assignment]
    observations: dict[str, np.ndarray]  # type: ignore[assignment]
    next_observations: dict[str, np.ndarray]  # type: ignore[assignment]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super(ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert isinstance(self.obs_shape, dict), "DictReplayBuffer must be used with Dict obs space only"
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        assert not optimize_memory_usage, "DictReplayBuffer does not support optimize_memory_usage"
        # disabling as this adds quite a bit of complexity
        # https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = {
            key: np.zeros((self.buffer_size, self.n_envs, *_obs_shape), dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }
        self.next_observations = {
            key: np.zeros((self.buffer_size, self.n_envs, *_obs_shape), dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            obs_nbytes = 0
            for _, obs in self.observations.items():
                obs_nbytes += obs.nbytes

            total_memory_usage: float = obs_nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            if not optimize_memory_usage:
                next_obs_nbytes = 0
                for _, obs in self.observations.items():
                    next_obs_nbytes += obs.nbytes
                total_memory_usage += next_obs_nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(  # type: ignore[override]
        self,
        obs: dict[str, np.ndarray],
        next_obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        # Copy to avoid modification by reference
        for key in self.observations.keys():
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = np.array(obs[key])

        for key in self.next_observations.keys():
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.next_observations[key][self.pos] = np.array(next_obs[key])

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(  # type: ignore[override]
        self,
        batch_size: int,
        env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        return super(ReplayBuffer, self).sample(batch_size=batch_size, env=env)

    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        # Normalize if needed and remove extra dimension (we are using only one env for now)
        obs_ = self._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()}, env)
        next_obs_ = self._normalize_obs(
            {key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()}, env
        )

        assert isinstance(obs_, dict)
        assert isinstance(next_obs_, dict)
        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(
                -1, 1
            ),
            rewards=self.to_torch(self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)),
        )


class DictRolloutBuffer(RolloutBuffer):
    """
    Dict Rollout buffer used in on-policy algorithms like A2C/PPO.
    Extends the RolloutBuffer to use dictionary observations

    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to Monte-Carlo advantage estimate when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observation_space: spaces.Dict
    obs_shape: dict[str, tuple[int, ...]]  # type: ignore[assignment]
    observations: dict[str, np.ndarray]  # type: ignore[assignment]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super(RolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert isinstance(self.obs_shape, dict), "DictRolloutBuffer must be used with Dict obs space only"

        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = {}
        for key, obs_input_shape in self.obs_shape.items():
            self.observations[key] = np.zeros(
                (self.buffer_size, self.n_envs, *obs_input_shape), dtype=self.observation_space[key].dtype
            )
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=self.action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def add(  # type: ignore[override]
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        for key in self.observations.keys():
            obs_ = np.array(obs[key])
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs_ = obs_.reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = obs_

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(  # type: ignore[override]
        self,
        batch_size: Optional[int] = None,
    ) -> Generator[DictRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            _tensor_names = ["actions", "values", "log_probs", "advantages", "returns"]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> DictRolloutBufferSamples:
        return DictRolloutBufferSamples(
            observations={key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations.items()},
            # Cast to float32 (backward compatible), this would lead to RuntimeError for MultiBinary space
            actions=self.to_torch(self.actions[batch_inds].astype(np.float32, copy=False)),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
        )


class NStepReplayBuffer(ReplayBuffer):
    """
    Replay buffer used for computing n-step returns in off-policy algorithms like SAC/DQN.

    The n-step return combines multiple steps of future rewards,
    discounted by the discount factor gamma.
    This can help improve sample efficiency and credit assignment.

    This implementation uses the same storage space as a normal replay buffer,
    and NumPy vectorized operations at sampling time to efficiently compute the
    n-step return, without requiring extra memory.

    This implementation is inspired by:
    - https://github.com/younggyoseo/FastTD3
    - https://github.com/DLR-RM/stable-baselines3/pull/81

    It avoids potential issues such as:
    - https://github.com/younggyoseo/FastTD3/issues/6

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Not supported
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    :param n_steps: Number of steps to accumulate rewards for n-step returns
    :param gamma: Discount factor for future rewards
    """

    def __init__(self, *args, n_steps: int = 3, gamma: float = 0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_steps = n_steps
        self.gamma = gamma
        if self.optimize_memory_usage:
            raise NotImplementedError("NStepReplayBuffer doesn't support optimize_memory_usage=True")

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample a batch of transitions and compute n-step returns.

        For each sampled transition, the method computes the cumulative discounted reward over
        the next `n_steps`, properly handling episode termination and timeouts.
        The next observation and done flag correspond to the last transition in the computed n-step trajectory.

        :param batch_inds: Indices of samples to retrieve
        :param env: Optional VecNormalize environment for normalizing observations/rewards
        :return: A batch of samples with n-step returns and corresponding observations/actions
        """
        # Randomly choose env indices for each sample
        env_indices = np.random.randint(0, self.n_envs, size=batch_inds.shape)

        # Note: the self.pos index is dangerous (will overlap two different episodes when buffer is full)
        # so we set self.pos-1 to truncated=True (temporarily) if done=False and truncated=False
        last_valid_index = self.pos - 1
        original_timeout_values = self.timeouts[last_valid_index].copy()
        self.timeouts[last_valid_index] = np.logical_or(original_timeout_values, np.logical_not(self.dones[last_valid_index]))

        # Compute n-step indices with wrap-around
        steps = np.arange(self.n_steps).reshape(1, -1)  # shape: [1, n_steps]
        indices = (batch_inds[:, None] + steps) % self.buffer_size  # shape: [batch, n_steps]

        # Retrieve sequences of transitions
        rewards_seq = self._normalize_reward(self.rewards[indices, env_indices[:, None]], env)  # [batch, n_steps]
        dones_seq = self.dones[indices, env_indices[:, None]]  # [batch, n_steps]
        truncated_seq = self.timeouts[indices, env_indices[:, None]]  # [batch, n_steps]

        # Compute masks: 1 until first done/truncation (inclusive)
        done_or_truncated = np.logical_or(dones_seq, truncated_seq)
        done_idx = done_or_truncated.argmax(axis=1)
        # If no done/truncation, keep full sequence
        has_done_or_truncated = done_or_truncated.any(axis=1)
        done_idx = np.where(has_done_or_truncated, done_idx, self.n_steps - 1)

        mask = np.arange(self.n_steps).reshape(1, -1) <= done_idx[:, None]  # shape: [batch, n_steps]
        # Compute discount factors for bootstrapping (using target Q-Value)
        # It is gamma ** n_steps by default but should be adjusted in case of early termination/truncation.
        target_q_discounts = self.gamma ** mask.sum(axis=1, keepdims=True).astype(np.float32)  # [batch, 1]

        # Apply discount
        discounts = self.gamma ** np.arange(self.n_steps, dtype=np.float32).reshape(1, -1)  # [1, n_steps]
        discounted_rewards = rewards_seq * discounts * mask
        n_step_returns = discounted_rewards.sum(axis=1, keepdims=True)  # [batch, 1]

        # Compute indices of next_obs/done at the final point of the n-step transition
        last_indices = (batch_inds + done_idx) % self.buffer_size
        next_obs = self._normalize_obs(self.next_observations[last_indices, env_indices], env)
        next_dones = self.dones[last_indices, env_indices][:, None].astype(np.float32)
        next_timeouts = self.timeouts[last_indices, env_indices][:, None].astype(np.float32)
        final_dones = next_dones * (1.0 - next_timeouts)

        # Revert back tmp changes to avoid sampling across episodes
        self.timeouts[last_valid_index] = original_timeout_values

        # Gather observations and actions
        obs = self._normalize_obs(self.observations[batch_inds, env_indices], env)
        actions = self.actions[batch_inds, env_indices]

        # SUFT: old values of the first transition of the n-step trajectory (the one being updated)
        if self.is_dqn:
            # DQN
            old_values = {
                "old_values": self.to_torch(self.old_values[batch_inds, env_indices]),
            }
        else:
            # SAC
            old_values = {
                "old_values1": self.to_torch(self.old_values1[batch_inds, env_indices]),
                "old_values2": self.to_torch(self.old_values2[batch_inds, env_indices]),
            }

        return ReplayBufferSamples(
            observations=self.to_torch(obs),  # type: ignore[arg-type]
            actions=self.to_torch(actions),
            next_observations=self.to_torch(next_obs),  # type: ignore[arg-type]
            dones=self.to_torch(final_dones),
            rewards=self.to_torch(n_step_returns),
            discounts=self.to_torch(target_q_discounts),
            **old_values,
        )


class ShortLongTermReplayBuffer(ReplayBuffer):
    """
    Double replay buffer made of a short term memory and a long term memory.

    Every new transition is added to the short term buffer (this buffer, of size ``short_term_buffer_size``).
    Once the short term buffer is full, every new transition pops out the oldest one
    (as in a standard replay buffer), and the popped transition is added to the
    long term buffer (of size ``long_term_buffer_size``), which is itself a standard replay buffer:
    once full, its oldest transition is discarded.

    :param short_term_buffer_size: Max number of element in the short term buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Not supported
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    :param long_term_buffer_size: Max number of element in the long term buffer
    :param sampling_fraction: Fraction of each batch taken from the short term buffer
        (the rest is taken from the long term buffer), see ``sample_short_and_long()``
    """

    def __init__(self,
                 short_term_buffer_size: int,
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 device: Union[th.device, str] = "auto",
                 n_envs: int = 1,
                 optimize_memory_usage: bool = False,
                 handle_timeout_termination: bool = True,
                 is_dqn: bool = False,
                 long_term_buffer_size: int = 1_000_000,
                 sampling_fraction: float = 0.5,
                 ):
        super().__init__(short_term_buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage, handle_timeout_termination, is_dqn)
        if self.optimize_memory_usage:
            raise NotImplementedError("ShortLongTermReplayBuffer doesn't support optimize_memory_usage=True")

        self.long_term_buffer = ReplayBuffer(
            long_term_buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
            is_dqn
        )
        self.sampling_fraction = sampling_fraction

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        old_values_to_store,
        infos: list[dict[str, Any]],
    ) -> None:
        if self.full:
            # add oldest sample from the short replay buffer
            self.long_term_buffer.add(*self._get_oldest())
        # add the regular sample into the short term buffer
        super().add(obs, next_obs, action, reward, done, old_values_to_store, infos)

    def _get_oldest(self) -> tuple:
        """
        Pull the oldest sample of the short term buffer (the one at ``self.pos`` once the buffer is full),
        in the format expected by ``add()``.
        """
        if self.is_dqn:
            # DQN
            old_values_to_store = th.as_tensor(self.old_values[self.pos])
        else:
            # SAC
            old_values_to_store = (th.as_tensor(self.old_values1[self.pos]), th.as_tensor(self.old_values2[self.pos]))
        infos = [{"TimeLimit.truncated": bool(timeout)} for timeout in self.timeouts[self.pos]]
        return (
            self.observations[self.pos],
            self.next_observations[self.pos],
            self.actions[self.pos],
            self.rewards[self.pos],
            self.dones[self.pos],
            old_values_to_store,
            infos,
        )

    def reset(self) -> None:
        super().reset()
        self.long_term_buffer.reset()

    def sample_short_and_long(
        self, batch_size: int, env: Optional[VecNormalize] = None
    ) -> tuple[Optional[ReplayBufferSamples], Optional[ReplayBufferSamples]]:
        """
        Sample elements from the short term and the long term buffers, according to a given ratio.
        As long as the long term buffer does not have enough samples,
        the whole batch is taken from the short term buffer.

        :param batch_size: Number of element to sample (short term + long term)
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: (short term samples, long term samples), one of them is ``None``
            when nothing is taken from the corresponding buffer
        """
        short_term_batch_size = round(batch_size * self.sampling_fraction)
        long_term_batch_size = batch_size - short_term_batch_size
        if not (self.long_term_buffer.pos >= long_term_batch_size or self.long_term_buffer.full):
            # Not enough samples in the long term buffer yet: take everything from the short term buffer
            short_term_batch_size, long_term_batch_size = batch_size, 0
        short_term_samples = self.sample(short_term_batch_size, env) if short_term_batch_size > 0 else None
        long_term_samples = self.long_term_buffer.sample(long_term_batch_size, env) if long_term_batch_size > 0 else None
        return short_term_samples, long_term_samples


class NStepLongTermReplayBuffer(ReplayBuffer):
    """
    Long term part of the ``NStepShortLongTermReplayBuffer``.

    Its transitions are already n-step transitions: they are computed by the short term buffer
    when the transition leaves it (at that moment its next ``n_steps - 1`` transitions are still in the short term buffer).
    Each stored row holds (s_t, a_t, R_t^(n), s_{t+k}, done_{t+k}, old values of s_t) and the bootstrap discount gamma^k,
    where k <= n_steps is the number of steps until the first termination/truncation.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Not supported
    :param handle_timeout_termination: Not used, the dones are already masked by the short term buffer
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.optimize_memory_usage:
            raise NotImplementedError("NStepLongTermReplayBuffer doesn't support optimize_memory_usage=True")
        # Bootstrap discount of each n-step transition: gamma ** k
        self.discounts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(  # type: ignore[override]
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        old_values_to_store,
        infos: list[dict[str, Any]],
        discount: np.ndarray,
    ) -> None:
        """
        :param reward: n-step return R_t^(n)
        :param done: true termination at the end of the n-step trajectory (timeouts already removed)
        :param discount: bootstrap discount gamma ** k
        """
        self.discounts[self.pos] = np.array(discount)
        super().add(obs, next_obs, action, reward, done, old_values_to_store, infos)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.is_dqn:
            # DQN
            old_values = {
                "old_values": self.to_torch(self.old_values[batch_inds, env_indices]),
            }
        else:
            # SAC
            old_values = {
                "old_values1": self.to_torch(self.old_values1[batch_inds, env_indices]),
                "old_values2": self.to_torch(self.old_values2[batch_inds, env_indices]),
            }
        return ReplayBufferSamples(
            observations=self.to_torch(self._normalize_obs(self.observations[batch_inds, env_indices, :], env)),
            actions=self.to_torch(self.actions[batch_inds, env_indices, :]),
            next_observations=self.to_torch(self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)),
            dones=self.to_torch(self.dones[batch_inds, env_indices].reshape(-1, 1)),
            # Note: with VecNormalize, the reward normalization is applied to the n-step return
            rewards=self.to_torch(self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)),
            discounts=self.to_torch(self.discounts[batch_inds, env_indices].reshape(-1, 1)),
            **old_values,
        )


class NStepShortLongTermReplayBuffer(NStepReplayBuffer):
    """
    N-step version of the ``ShortLongTermReplayBuffer``.

    Every new transition is added to the short term buffer (this buffer, of size ``short_term_buffer_size``),
    which is an ``NStepReplayBuffer``: the n-step returns of its samples are computed at sampling time
    (truncated at the newest transition, whose next transitions were not collected yet).
    Once the short term buffer is full, every new transition pops out the oldest one.
    At that moment its next ``n_steps - 1`` transitions are still in the short term buffer, so its full n-step
    transition (n-step return, s_{t+k}, done, gamma^k) is computed and added to the long term buffer
    (``NStepLongTermReplayBuffer``, of size ``long_term_buffer_size``). The n-step returns of the long term samples
    are therefore never cut at the border between the two buffers.
    The old values (SUFT) of s_t move with the transition.

    :param short_term_buffer_size: Max number of element in the short term buffer (must be >= ``n_steps``)
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Not supported
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    :param long_term_buffer_size: Max number of element in the long term buffer
    :param sampling_fraction: Fraction of each batch taken from the short term buffer
        (the rest is taken from the long term buffer), see ``sample_short_and_long()``
    :param n_steps: Number of steps to accumulate rewards for n-step returns
    :param gamma: Discount factor for future rewards
    """

    def __init__(
        self,
        short_term_buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        is_dqn: bool = False,
        long_term_buffer_size: int = 1_000_000,
        sampling_fraction: float = 0.5,
        n_steps: int = 3,
        gamma: float = 0.99,
    ):
        super().__init__(
            short_term_buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
            is_dqn,
            n_steps=n_steps,
            gamma=gamma,
        )
        assert self.buffer_size >= n_steps, "The short term buffer must hold at least `n_steps` transitions per env"

        self.long_term_buffer = NStepLongTermReplayBuffer(
            long_term_buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
            is_dqn,
        )
        self.sampling_fraction = sampling_fraction

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        old_values_to_store,
        infos: list[dict[str, Any]],
    ) -> None:
        if self.full:
            # add the n-step transition of the oldest sample of the short replay buffer
            self.long_term_buffer.add(*self._get_oldest_n_step())
        # add the regular sample into the short term buffer
        super().add(obs, next_obs, action, reward, done, old_values_to_store, infos)

    def _get_oldest_n_step(self) -> tuple:
        """
        Build the n-step transition of the oldest sample of the short term buffer (the one at ``self.pos``
        once the buffer is full), for every env, in the format expected by ``NStepLongTermReplayBuffer.add()``.
        Its next ``n_steps - 1`` transitions are the next rows of the short term buffer (all already collected).
        """
        env_indices = np.arange(self.n_envs)
        # Indices of the n-step trajectory, shape: [n_steps]
        indices = (self.pos + np.arange(self.n_steps)) % self.buffer_size
        rewards_seq = self.rewards[indices].T  # [n_envs, n_steps]
        dones_seq = self.dones[indices].T
        truncated_seq = self.timeouts[indices].T

        # Stop at the first done/truncation (inclusive), otherwise keep the full sequence
        done_or_truncated = np.logical_or(dones_seq, truncated_seq)
        done_idx = np.where(done_or_truncated.any(axis=1), done_or_truncated.argmax(axis=1), self.n_steps - 1)  # [n_envs]
        mask = np.arange(self.n_steps).reshape(1, -1) <= done_idx[:, None]  # [n_envs, n_steps]

        discounts = self.gamma ** np.arange(self.n_steps, dtype=np.float32).reshape(1, -1)  # [1, n_steps]
        n_step_returns = (rewards_seq * discounts * mask).sum(axis=1).astype(np.float32)  # [n_envs]
        # Bootstrap discount: gamma ** k, k = number of rewards in the n-step return
        target_q_discounts = (self.gamma ** mask.sum(axis=1)).astype(np.float32)  # [n_envs]

        # next_obs/done at the final point of the n-step trajectory (only true terminations stop the bootstrap)
        last_indices = (self.pos + done_idx) % self.buffer_size
        next_obs = self.next_observations[last_indices, env_indices]
        final_dones = self.dones[last_indices, env_indices] * (1.0 - self.timeouts[last_indices, env_indices])

        # SUFT: old values of s_t
        if self.is_dqn:
            # DQN
            old_values_to_store = th.as_tensor(self.old_values[self.pos])
        else:
            # SAC
            old_values_to_store = (th.as_tensor(self.old_values1[self.pos]), th.as_tensor(self.old_values2[self.pos]))
        # The timeouts are already handled in final_dones
        infos = [{"TimeLimit.truncated": False} for _ in range(self.n_envs)]
        return (
            self.observations[self.pos],
            next_obs,
            self.actions[self.pos],
            n_step_returns,
            final_dones,
            old_values_to_store,
            infos,
            target_q_discounts,
        )

    def reset(self) -> None:
        super().reset()
        self.long_term_buffer.reset()

    def sample_short_and_long(
        self, batch_size: int, env: Optional[VecNormalize] = None
    ) -> tuple[Optional[ReplayBufferSamples], Optional[ReplayBufferSamples]]:
        """
        Sample n-step transitions from the short term and the long term buffers, according to a given ratio.
        As long as the long term buffer does not have enough samples,
        the whole batch is taken from the short term buffer.

        :param batch_size: Number of element to sample (short term + long term)
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: (short term samples, long term samples), one of them is ``None``
            when nothing is taken from the corresponding buffer
        """
        short_term_batch_size = round(batch_size * self.sampling_fraction)
        long_term_batch_size = batch_size - short_term_batch_size
        if not (self.long_term_buffer.pos >= long_term_batch_size or self.long_term_buffer.full):
            # Not enough samples in the long term buffer yet: take everything from the short term buffer
            short_term_batch_size, long_term_batch_size = batch_size, 0
        short_term_samples = self.sample(short_term_batch_size, env) if short_term_batch_size > 0 else None
        long_term_samples = self.long_term_buffer.sample(long_term_batch_size, env) if long_term_batch_size > 0 else None
        return short_term_samples, long_term_samples


class PrioritySumTree:
    """
    Binary sum tree over the priorities of the ``PrioritizedReplayBuffer``, vectorized with NumPy:
    O(log N) updates and prefix-sum searches for a whole batch at once.
    Same role as Dopamine's ``SumTree`` (dopamine/jax/replay_memory/sum_tree.py), including ``max_recorded_priority``.

    Leaf ``i`` holds the priority of the flat transition index ``i``, empty leaves hold 0.

    :param capacity: Number of leaves (rounded up to a power of 2)
    """

    def __init__(self, capacity: int):
        self.capacity = 1 << int(np.ceil(np.log2(max(capacity, 2))))
        self.sum_tree = np.zeros(2 * self.capacity, dtype=np.float64)
        # Largest priority ever set, given to the new transitions (Dopamine: starts at 1.0)
        self.max_recorded_priority = 1.0

    @property
    def root(self) -> float:
        """The root value (total sum) of the sum tree."""
        return float(self.sum_tree[1])

    def get(self, leaves: np.ndarray) -> np.ndarray:
        return self.sum_tree[np.asarray(leaves) + self.capacity]

    def set(self, leaves: np.ndarray, values: np.ndarray) -> None:
        """
        Set the value of the given leaves and update their ancestors (all leaves are at the same depth).
        """
        values = np.asarray(values, dtype=np.float64)
        self.max_recorded_priority = max(self.max_recorded_priority, float(values.max()))
        # De-duplicate the leaves, keeping the first value of each (as Dopamine)
        leaves, first = np.unique(np.asarray(leaves, dtype=np.int64), return_index=True)
        nodes = leaves + self.capacity
        self.sum_tree[nodes] = values[first]
        nodes = np.unique(nodes // 2)
        while True:
            self.sum_tree[nodes] = self.sum_tree[2 * nodes] + self.sum_tree[2 * nodes + 1]
            if nodes[0] == 1:
                break
            nodes = np.unique(nodes // 2)

    def query(self, targets: np.ndarray) -> np.ndarray:
        """
        For every target v in [0, root), find the leaf i such that sum_{j<i} p_j <= v < sum_{j<=i} p_j.
        Never returns an empty (zero priority) leaf as long as the root is positive.
        """
        targets = np.array(targets, dtype=np.float64)
        nodes = np.ones(len(targets), dtype=np.int64)
        while nodes[0] < self.capacity:
            left = 2 * nodes
            left_sum = self.sum_tree[left]
            # Go right when v is past the left subtree, unless the right subtree is empty (float rounding at the end)
            go_right = (targets >= left_sum) & (self.sum_tree[left + 1] > 0)
            targets -= left_sum * go_right
            nodes = left + go_right
        return nodes - self.capacity

    def stratified_sample(self, batch_size: int) -> np.ndarray:
        """
        Divide [0, root) into ``batch_size`` equal segments, pick a random number from each segment
        and use it to sample from the sum tree (as specified in Schaul et al. (2016), Dopamine TF ``stratified_sample``).
        """
        if self.root == 0.0:
            raise ValueError("Cannot sample from an empty sum tree.")
        bounds = np.linspace(0.0, 1.0, batch_size + 1)
        query_values = np.random.uniform(bounds[:-1], bounds[1:]) * self.root
        return self.query(query_values)


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay buffer (proportional variant), adapted from Google Dopamine
    (dopamine/jax/replay_memory/samplers.py ``PrioritizedSamplingDistribution``,
    dopamine/tf/replay_memory/prioritized_replay_buffer.py and sum_tree.py).
    Paper: Schaul et al. (2016), https://arxiv.org/abs/1511.05952

    - Transition i is sampled with probability P(i) = p_i^priority_exponent / sum_k p_k^priority_exponent,
      with stratified sampling over the sum tree.
    - New transitions get the maximal priority recorded so far (``max_recorded_priority``, 1.0 at the start).
    - ``sample()`` returns the sampling probabilities P(i) and the indices of the sampled transitions;
      the agent computes the importance-sampling loss weights and sets the new priorities with ``update_priorities()``
      (in Dopamine: loss_weights = 1 / sqrt(P(i) + 1e-10) normalized by their max over the batch,
      and priorities = sqrt(per-sample loss + 1e-10)).

    The priorities are kept in a sum tree over the flat indices ``pos * n_envs + env``.
    The old values (SUFT) are stored and returned as in the ``ReplayBuffer``.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Not supported
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    :param priority_exponent: Exponent applied to the priorities given to ``update_priorities()``
        (Dopamine default 1.0: the agent already takes the square root of the loss)
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        is_dqn: bool = False,
        priority_exponent: float = 1.0,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage, handle_timeout_termination, is_dqn)
        if self.optimize_memory_usage:
            raise NotImplementedError("PrioritizedReplayBuffer doesn't support optimize_memory_usage=True")
        self.priority_exponent = priority_exponent
        self.sum_tree = PrioritySumTree(self.buffer_size * self.n_envs)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        old_values_to_store,
        infos: list[dict[str, Any]],
    ) -> None:
        # Schaul et al.'s scheme: the new transitions (one per env, overwriting the oldest ones when full)
        # get the maximum priority so far
        flat_indices = self.pos * self.n_envs + np.arange(self.n_envs)
        self.sum_tree.set(flat_indices, np.full(self.n_envs, self.sum_tree.max_recorded_priority))
        super().add(obs, next_obs, action, reward, done, old_values_to_store, infos)

    def reset(self) -> None:
        super().reset()
        self.sum_tree = PrioritySumTree(self.buffer_size * self.n_envs)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> PrioritizedReplayBufferSamples:  # type: ignore[override]
        """
        Sample a prioritized batch (stratified sampling), with its sampling probabilities and its indices.

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        flat_indices = self.sum_tree.stratified_sample(batch_size)
        sampling_probabilities = self.sum_tree.get(flat_indices) / self.sum_tree.root
        return self._get_prioritized_samples(flat_indices, sampling_probabilities.astype(np.float32), env)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Set the priorities of the sampled transitions (after the update), as Dopamine ``update()``:
        p_i <- priority_i ** priority_exponent (a zero priority stays zero).

        :param indices: Flat indices returned with the samples
        :param priorities: New priorities of the sampled transitions
        """
        priorities = np.asarray(priorities, dtype=np.float64).flatten()
        priorities = np.where(priorities == 0.0, 0.0, priorities**self.priority_exponent)
        self.sum_tree.set(indices, priorities)

    def _get_prioritized_samples(
        self, flat_indices: np.ndarray, sampling_probabilities: np.ndarray, env: Optional[VecNormalize] = None
    ) -> PrioritizedReplayBufferSamples:
        batch_inds, env_indices = flat_indices // self.n_envs, flat_indices % self.n_envs

        if self.is_dqn:
            # DQN
            old_values = {
                "old_values": self.to_torch(self.old_values[batch_inds, env_indices]),
            }
        else:
            # SAC
            old_values = {
                "old_values1": self.to_torch(self.old_values1[batch_inds, env_indices]),
                "old_values2": self.to_torch(self.old_values2[batch_inds, env_indices]),
            }
        return PrioritizedReplayBufferSamples(
            observations=self.to_torch(self._normalize_obs(self.observations[batch_inds, env_indices, :], env)),
            actions=self.to_torch(self.actions[batch_inds, env_indices, :]),
            next_observations=self.to_torch(self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)),
            # Only use dones that are not due to timeouts
            dones=self.to_torch((self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1)),
            rewards=self.to_torch(self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)),
            sampling_probabilities=self.to_torch(sampling_probabilities.reshape(-1, 1)),
            indices=flat_indices,
            **old_values,
        )


class ShortUniformLongPrioritizedReplayBuffer(ShortLongTermReplayBuffer):
    """
    ``ShortLongTermReplayBuffer`` whose short term buffer (this buffer) is sampled uniformly
    and whose long term buffer is a ``PrioritizedReplayBuffer``.

    Every new transition is added to the short term buffer. Once the short term buffer is full, every new transition
    pops out the oldest one, which is added to the long term buffer. The short term transitions have no priority,
    so a transition entering the long term buffer gets its maximal recorded priority (Schaul et al.'s scheme
    for new transitions).
    The long term samples come with their sampling probabilities and indices, the agent updates their priorities
    with ``long_term_buffer.update_priorities()``.

    :param short_term_buffer_size: Max number of element in the short term buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Not supported
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    :param long_term_buffer_size: Max number of element in the long term buffer
    :param sampling_fraction: Fraction of each batch taken from the short term buffer
        (the rest is taken from the long term buffer), see ``sample_short_and_long()``
    :param priority_exponent: Exponent applied to the priorities of the long term buffer
    """

    def __init__(
        self,
        short_term_buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        is_dqn: bool = False,
        long_term_buffer_size: int = 1_000_000,
        sampling_fraction: float = 0.5,
        priority_exponent: float = 1.0,
    ):
        # The short term part is a plain ReplayBuffer (ShortLongTermReplayBuffer.__init__ is not called
        # as it would also allocate a uniform long term buffer)
        ReplayBuffer.__init__(
            self, short_term_buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage, handle_timeout_termination, is_dqn
        )
        if self.optimize_memory_usage:
            raise NotImplementedError("ShortUniformLongPrioritizedReplayBuffer doesn't support optimize_memory_usage=True")

        self.long_term_buffer = PrioritizedReplayBuffer(
            long_term_buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
            is_dqn,
            priority_exponent=priority_exponent,
        )
        self.sampling_fraction = sampling_fraction


class ShortLongTermPrioritizedReplayBuffer(PrioritizedReplayBuffer):
    """
    Short term and long term replay buffer where both buffers are ``PrioritizedReplayBuffer``.

    Every new transition is added to the short term buffer (this buffer) with its maximal recorded priority.
    Once the short term buffer is full, every new transition pops out the oldest one, which is added to the
    long term buffer with the priority it had in the short term buffer (its last TD error based priority,
    or the max priority if it was never sampled).
    Both parts of a batch come with their sampling probabilities and indices, the agent updates the priorities
    with ``update_priorities()`` (short term) and ``long_term_buffer.update_priorities()`` (long term).

    :param short_term_buffer_size: Max number of element in the short term buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Not supported
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    :param long_term_buffer_size: Max number of element in the long term buffer
    :param sampling_fraction: Fraction of each batch taken from the short term buffer
        (the rest is taken from the long term buffer), see ``sample_short_and_long()``
    :param priority_exponent: Exponent applied to the priorities of both buffers
    """

    def __init__(
        self,
        short_term_buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        is_dqn: bool = False,
        long_term_buffer_size: int = 1_000_000,
        sampling_fraction: float = 0.5,
        priority_exponent: float = 1.0,
    ):
        super().__init__(
            short_term_buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
            is_dqn,
            priority_exponent=priority_exponent,
        )
        self.long_term_buffer = PrioritizedReplayBuffer(
            long_term_buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
            is_dqn,
            priority_exponent=priority_exponent,
        )
        self.sampling_fraction = sampling_fraction

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        old_values_to_store,
        infos: list[dict[str, Any]],
    ) -> None:
        if self.full:
            # Move the oldest sample of the short term buffer to the long term buffer, with its priority
            # (the priorities are stored in the sum trees after the exponent, which is the same for both buffers)
            env_indices = np.arange(self.n_envs)
            priorities = self.sum_tree.get(self.pos * self.n_envs + env_indices)
            long_term_pos = self.long_term_buffer.pos
            self.long_term_buffer.add(*ShortLongTermReplayBuffer._get_oldest(self))
            self.long_term_buffer.sum_tree.set(long_term_pos * self.n_envs + env_indices, priorities)
        # add the regular sample into the short term buffer (with the max recorded priority)
        super().add(obs, next_obs, action, reward, done, old_values_to_store, infos)

    def reset(self) -> None:
        super().reset()
        self.long_term_buffer.reset()

    def sample_short_and_long(
        self, batch_size: int, env: Optional[VecNormalize] = None
    ) -> tuple[Optional[PrioritizedReplayBufferSamples], Optional[PrioritizedReplayBufferSamples]]:
        """
        Sample prioritized elements from the short term and the long term buffers, according to a given ratio
        (same rule as ``ShortLongTermReplayBuffer.sample_short_and_long()``).

        :param batch_size: Number of element to sample (short term + long term)
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: (short term samples, long term samples), one of them is ``None``
            when nothing is taken from the corresponding buffer
        """
        return ShortLongTermReplayBuffer.sample_short_and_long(self, batch_size, env)  # type: ignore[arg-type,return-value]


class CombinedReplayBuffer(ReplayBuffer):
    """
    Combined Experience Replay (CER)
    Paper: Zhang & Sutton (2017), A Deeper Look at Experience Replay, https://arxiv.org/abs/1712.01275

    Every sampled minibatch contains the latest transition, the rest of the minibatch is sampled uniformly.
    This makes the agent learn from every new transition right away, and reduces the sensitivity to the replay
    buffer size.
    The uniform part is sampled from the older transitions only, so that the latest transitions are not sampled twice
    in the same minibatch (the paper samples it from the whole buffer, the difference is ``n_latest`` of N rows).
    As in every buffer of this repo, the uniform part is sampled with replacement.

    ``n_latest`` generalizes it to the latest ``n_latest`` transitions (1 in the paper, where the agent is updated
    after every step). With ``n_latest = train_freq``, every batch contains all the transitions collected since
    the previous update, so every transition is replayed at least once even when updating every ``train_freq`` steps
    (see ``n_latest_from_train_freq()``).
    With several envs, the latest ``n_latest`` transitions of every env are added (``n_latest * n_envs`` transitions),
    and the rest of the minibatch is sampled uniformly from the older transitions.
    The old values (SUFT) are stored and returned as in the ``ReplayBuffer``.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Not supported
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    :param n_latest: Number of latest transitions (per env) added to every minibatch (1 in the paper)
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        is_dqn: bool = False,
        n_latest: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage, handle_timeout_termination, is_dqn)
        if self.optimize_memory_usage:
            raise NotImplementedError("CombinedReplayBuffer doesn't support optimize_memory_usage=True")
        assert 1 <= n_latest <= self.buffer_size, f"n_latest must be in [1, buffer size per env], got {n_latest}"
        self.n_latest = n_latest

    @staticmethod
    def n_latest_from_train_freq(train_freq: Union[int, tuple[int, str]]) -> int:
        """
        Number of transitions (per env) collected between two updates: ``train_freq`` when it is given in steps.
        With a frequency in episodes the number of transitions varies, so only the latest one is used.

        :param train_freq: The ``train_freq`` argument of the agent (``int`` or ``(frequency, unit)``)
        :return: The ``n_latest`` that adds all the transitions collected since the previous update to every batch
        """
        frequency, unit = (train_freq, "step") if isinstance(train_freq, int) else train_freq
        if unit != "step":
            warnings.warn(f"CER: train_freq={train_freq} is not in steps, only the latest transition is added to every batch")
            return 1
        return int(frequency)

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample the latest ``n_latest`` transitions (of every env), newest first,
        and the rest of the minibatch uniformly from the older transitions.

        :param batch_size: Number of element to sample (latest transitions included)
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        # The latest n_latest rows (only the ones already collected), newest first, every env of each row
        n_rows = min(self.n_latest, upper_bound)
        n_latest_samples = n_rows * self.n_envs
        assert batch_size >= n_latest_samples, (
            f"The batch size ({batch_size}) must be at least the number of latest transitions ({n_latest_samples})"
        )
        rows = (self.pos - 1 - np.arange(n_rows)) % self.buffer_size
        latest_inds = np.repeat(rows, self.n_envs)
        latest_env_indices = np.tile(np.arange(self.n_envs), n_rows)
        # Uniform samples for the rest of the minibatch, from the older rows only so that the latest transitions
        # are not sampled twice (unless the buffer holds nothing else yet)
        n_older_rows = upper_bound - n_rows
        if n_older_rows > 0:
            # The older rows, from the oldest one (self.pos when the buffer is full, 0 otherwise)
            oldest_row = self.pos if self.full else 0
            uniform_inds = (oldest_row + np.random.randint(0, n_older_rows, size=batch_size - n_latest_samples)) % self.buffer_size
        else:
            uniform_inds = np.random.randint(0, upper_bound, size=batch_size - n_latest_samples)
        uniform_env_indices = np.random.randint(0, high=self.n_envs, size=batch_size - n_latest_samples)
        batch_inds = np.concatenate([latest_inds, uniform_inds])
        env_indices = np.concatenate([latest_env_indices, uniform_env_indices])
        return self._get_combined_samples(batch_inds, env_indices, env)

    def _get_combined_samples(
        self, batch_inds: np.ndarray, env_indices: np.ndarray, env: Optional[VecNormalize] = None
    ) -> ReplayBufferSamples:
        # Same as ReplayBuffer._get_samples(), with the given env indices
        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env),
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        if self.is_dqn:
            # DQN
            old_values = {
                "old_values": self.to_torch(self.old_values[batch_inds, env_indices]),
            }
        else:
            # SAC
            old_values = {
                "old_values1": self.to_torch(self.old_values1[batch_inds, env_indices]),
                "old_values2": self.to_torch(self.old_values2[batch_inds, env_indices]),
            }
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)), **old_values)


class ShortLongTermCombinedReplayBuffer(ShortLongTermReplayBuffer, CombinedReplayBuffer):
    """
    ``ShortLongTermReplayBuffer`` with Combined Experience Replay (Zhang & Sutton (2017), https://arxiv.org/abs/1712.01275):
    both buffers are sampled uniformly, and the latest transition is added to every batch.

    Every new transition is added to the short term buffer (this buffer), and once it is full the oldest one
    moves to the long term buffer (a uniform ``ReplayBuffer``), as in the ``ShortLongTermReplayBuffer``.
    The latest transitions are always in the short term buffer: the short term part of every batch is sampled
    as in the ``CombinedReplayBuffer`` (the latest ``n_latest`` transitions of every env, the rest uniformly),
    and the long term part uniformly.

    :param short_term_buffer_size: Max number of element in the short term buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Not supported
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    :param long_term_buffer_size: Max number of element in the long term buffer
    :param sampling_fraction: Fraction of each batch taken from the short term buffer
        (the rest is taken from the long term buffer), see ``sample_short_and_long()``
    :param n_latest: Number of latest transitions (per env) added to every minibatch (1 in the paper),
        at most the short term buffer size
    """

    # add() and reset() are the ones of ShortLongTermReplayBuffer, sample() is the one of CombinedReplayBuffer
    # (latest transitions + uniform)

    def __init__(
        self,
        short_term_buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        is_dqn: bool = False,
        long_term_buffer_size: int = 1_000_000,
        sampling_fraction: float = 0.5,
        n_latest: int = 1,
    ):
        # The short term part is initialized by CombinedReplayBuffer.__init__ (next in the MRO), the long term part
        # is a uniform ReplayBuffer
        ShortLongTermReplayBuffer.__init__(
            self,
            short_term_buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
            is_dqn,
            long_term_buffer_size=long_term_buffer_size,
            sampling_fraction=sampling_fraction,
        )
        # The latest transitions must all be in the short term buffer
        assert 1 <= n_latest <= self.buffer_size, f"n_latest must be in [1, short term buffer size per env], got {n_latest}"
        self.n_latest = n_latest

    def sample_short_and_long(
        self, batch_size: int, env: Optional[VecNormalize] = None
    ) -> tuple[Optional[ReplayBufferSamples], Optional[ReplayBufferSamples]]:
        """
        Sample elements from the short term and the long term buffers, according to a given ratio
        (same rule as ``ShortLongTermReplayBuffer.sample_short_and_long()``), with the latest ``n_latest`` transitions
        (of every env) in the short term part of every batch.
        The short term part has at least ``n_latest * n_envs`` elements so that it always holds the latest transitions.

        :param batch_size: Number of element to sample (short term + long term)
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: (short term samples, long term samples), the long term samples are ``None``
            when nothing is taken from the long term buffer
        """
        short_term_batch_size = max(round(batch_size * self.sampling_fraction), self.n_latest * self.n_envs)
        long_term_batch_size = batch_size - short_term_batch_size
        if not (self.long_term_buffer.pos >= long_term_batch_size or self.long_term_buffer.full):
            # Not enough samples in the long term buffer yet: take everything from the short term buffer
            short_term_batch_size, long_term_batch_size = batch_size, 0
        # CombinedReplayBuffer.sample(): the latest transitions + uniform short term samples
        short_term_samples = self.sample(short_term_batch_size, env)
        long_term_samples = self.long_term_buffer.sample(long_term_batch_size, env) if long_term_batch_size > 0 else None
        return short_term_samples, long_term_samples


def describe_replay_buffer(replay_buffer: ReplayBuffer) -> str:
    """
    Human readable description of a replay buffer: its kind (uniform / prioritized / n-step / combined) and its size,
    and for the short term + long term buffers the description of both parts and the sampling fraction.

    :param replay_buffer: The replay buffer of an agent
    :return: The description (one line, or three lines for a short term + long term buffer)
    """

    def describe(buffer: ReplayBuffer) -> str:
        if isinstance(buffer, PrioritizedReplayBuffer):
            kind = f"prioritized sampling (proportional PER, priority exponent {buffer.priority_exponent})"
        elif isinstance(buffer, NStepReplayBuffer):
            kind = f"uniform sampling, {buffer.n_steps}-step returns (gamma {buffer.gamma})"
        elif isinstance(buffer, NStepLongTermReplayBuffer):
            kind = "uniform sampling, n-step transitions"
        elif isinstance(buffer, CombinedReplayBuffer):
            kind = f"combined experience replay (latest {buffer.n_latest} transition(s) per env in every batch, the rest sampled uniformly)"
        else:
            kind = "uniform sampling"
        return f"{kind}, size {buffer.buffer_size * buffer.n_envs}"

    long_term_buffer = getattr(replay_buffer, "long_term_buffer", None)
    if long_term_buffer is None:
        return f"{type(replay_buffer).__name__}: {describe(replay_buffer)}"
    return (
        f"{type(replay_buffer).__name__}: short term + long term buffers, "
        f"{replay_buffer.sampling_fraction} of each batch from the short term buffer\n"  # type: ignore[attr-defined]
        f"  short term buffer: {describe(replay_buffer)}\n"
        f"  long term buffer: {type(long_term_buffer).__name__}: {describe(long_term_buffer)}"
    )
