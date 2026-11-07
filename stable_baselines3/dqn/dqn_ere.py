from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.buffers import EREReplayBuffer, ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, ReplayBufferSamples, Schedule
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.dqn.policies import DQNPolicy

SelfDQNERE = TypeVar("SelfDQNERE", bound="DQNERE")


class DQNERE(DQN):
    """
    Deep Q-Network (Vanilla DQN 2015, with target network) with Emphasizing Recent Experience (ERE):
    Wang, Wu, Vuong & Ross (2020), Striving for Simplicity and Performance in Off-Policy DRL: Output Normalization
    and Non-Uniform Sampling, ICML, https://proceedings.mlr.press/v119/wang20x.html

    ERE phase (after every episode, as in the paper and the authors' code): K = ``ere_updates_per_step`` * (number of
    environment steps since the previous phase) updates, the k-th one sampling uniformly from the most recent

        c_k = max(N * eta^(k * 1000 / K), c_min)

    transitions (see ``EREReplayBuffer``): the first update samples from the whole buffer, the last ones from the most
    recent data. eta is linearly annealed from ``ere_eta0`` to 1 (uniform) over the training.

    Two update schedules:
    - ``ere_continuous_updates=True`` (default): the usual DQN updates (``gradient_steps`` every ``train_freq`` steps,
      replay ratio 0.25 by default, uniform sampling, SUFT term as in ``DQN``) plus the ERE phase after every episode
      (the total number of updates is the sum of both)
    - ``ere_continuous_updates=False`` (paper): the agent is only updated in the ERE phases
      (``ere_updates_per_step = 0.25`` keeps the number of updates of DQN, 1 every 4 steps)
    A phase starts at the first training call after the end of an episode (at most ``train_freq - 1`` steps later).

    Loss of every update (as ``DQN``):
        loss = mean_j (Q(s_j, a_j) - y_j)^2 + tf_lambda * mean_j (Q(s_j, a_j) - Q_old(s_j, a_j))^2,
        y_j = r_j + gamma * (1 - done_j) * max_a Q_target(s'_j, a)
    with the SUFT term when ``tf_lambda > 0``. Batch size 32, learning rate 1e-4, ... as the ``DQN`` of this repo.

    :param ere_eta0: Initial eta, linearly annealed to 1 over the training (0.995 in the authors' code, 1: uniform)
    :param ere_c_min: Minimum number of recent transitions to sample from (5000 in the paper)
    :param ere_updates_per_step: Updates of an ERE phase per environment step since the previous phase
    :param ere_continuous_updates: Also do the usual DQN updates every ``train_freq`` steps (default),
        otherwise update only in the ERE phases (paper)
    See ``DQN`` for the other parameters.
    """

    # Replay buffer of the agent (the short / long term agent uses its short term + long term ERE version)
    replay_buffer_cls: ClassVar[type[ReplayBuffer]] = EREReplayBuffer
    replay_buffer: EREReplayBuffer

    def __init__(
        self,
        policy: Union[str, type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        ere_eta0: float = 0.995,
        ere_c_min: int = 5000,
        ere_updates_per_step: float = 0.25,
        ere_continuous_updates: bool = True,
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        target_update_interval: int = 10000,
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
    ) -> None:
        assert 0.0 < ere_eta0 <= 1.0, f"ere_eta0 must be in (0, 1], got {ere_eta0}"
        replay_buffer_kwargs = {"c_min": ere_c_min, **(replay_buffer_kwargs or {})}
        self.ere_eta0 = ere_eta0
        self.ere_updates_per_step = ere_updates_per_step
        self.ere_continuous_updates = ere_continuous_updates
        # Episode count and timestep of the last ERE phase
        self._ere_last_episode = 0
        self._ere_last_timestep = 0
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            replay_buffer_class=self.replay_buffer_cls,
            replay_buffer_kwargs=replay_buffer_kwargs,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
            tf_lambda=tf_lambda,
        )

    # ------------------------------------------------------------------------------------------------------
    # Sampling and loss (overridden by the short / long term agent)
    # ------------------------------------------------------------------------------------------------------
    def _ere_buffer(self) -> EREReplayBuffer:
        """The buffer whose recent transitions are emphasized."""
        return self.replay_buffer

    def _sample_uniform(self, batch_size: int) -> Any:
        return self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

    def _sample_recent(self, batch_size: int, c_k: int) -> Any:
        return self.replay_buffer.sample_recent(batch_size, c_k, env=self._vec_normalize_env)

    def _dqn_loss(self, replay_data: ReplayBufferSamples, suft: bool, reduction: str = "mean") -> th.Tensor:
        """DQN TD loss (target network), plus the SUFT term if ``suft`` and ``tf_lambda > 0``."""
        with th.no_grad():
            # Compute the next Q-values using the target network, greedy policy
            next_q_values, _ = self.q_net_target(replay_data.next_observations).max(dim=1)
            # 1-step TD target
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values.reshape(-1, 1)
        # Get current Q-values estimates for the actions from the replay buffer
        current_q_values = th.gather(self.q_net(replay_data.observations), dim=1, index=replay_data.actions.long())
        # SUFT CHANGE
        loss = F.mse_loss(current_q_values, target_q_values, reduction=reduction)
        if suft and self.tf_lambda > 0:
            old_q_values = th.gather(replay_data.old_values, dim=1, index=replay_data.actions.long())
            loss = loss + F.mse_loss(current_q_values, old_q_values, reduction=reduction) * self.tf_lambda
        # SUFT CHANGE until here
        return loss

    def _loss(self, data: Any, batch_size: int) -> th.Tensor:
        # SUFT term on every sample
        return self._dqn_loss(data, suft=True)

    # ------------------------------------------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------------------------------------------
    def _gradient_step(self, data: Any, batch_size: int) -> float:
        loss = self._loss(data, batch_size)
        # Optimize the policy
        self.policy.optimizer.zero_grad()
        loss.backward()
        # Clip gradient norm
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()
        return loss.item()

    def _ere_phase_size(self) -> int:
        """
        Number of updates K of the ERE phase to run now: 0 if no episode ended since the previous phase,
        otherwise ``ere_updates_per_step`` * (environment steps since the previous phase, after ``learning_starts``).
        """
        if self._episode_num < self._ere_last_episode or self.num_timesteps < self._ere_last_timestep:
            # learn() was called again with reset_num_timesteps=True
            self._ere_last_episode, self._ere_last_timestep = 0, 0
        if self._episode_num == self._ere_last_episode:
            return 0
        n_steps = self.num_timesteps - max(self._ere_last_timestep, self.learning_starts)
        self._ere_last_episode, self._ere_last_timestep = self._episode_num, self.num_timesteps
        return max(1, int(round(n_steps * self.ere_updates_per_step)))

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses, n_updates = [], 0
        if self.ere_continuous_updates:
            # Usual DQN updates (uniform sampling)
            for _ in range(gradient_steps):
                losses.append(self._gradient_step(self._sample_uniform(batch_size), batch_size))
            n_updates += gradient_steps

        n_ere_updates = self._ere_phase_size()
        if n_ere_updates > 0:
            # Linearly anneal eta from eta_0 to 1 over the training
            # (clipped: the last episode can end after total_timesteps, i.e. a negative progress remaining)
            progress = min(1.0, max(0.0, 1.0 - self._current_progress_remaining))
            eta = self.ere_eta0 + (1.0 - self.ere_eta0) * progress
            windows = []
            for k in range(n_ere_updates):
                # ERE: uniformly from the c_k most recent transitions
                c_k = self._ere_buffer().recent_window(k, n_ere_updates, eta)
                windows.append(c_k)
                losses.append(self._gradient_step(self._sample_recent(batch_size, c_k), batch_size))
            n_updates += n_ere_updates
            self.logger.record("train/ere_eta", eta)
            self.logger.record("train/ere_n_updates", n_ere_updates)
            self.logger.record("train/ere_c_k_first", windows[0])
            self.logger.record("train/ere_c_k_last", windows[-1])

        if n_updates == 0:
            return
        # Increase update counter
        self._n_updates += n_updates

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def learn(
        self: SelfDQNERE,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DQNERE",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDQNERE:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
