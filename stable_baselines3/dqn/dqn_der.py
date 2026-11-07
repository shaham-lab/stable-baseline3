from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th

from stable_baselines3.common.buffers import DER_H2_FRACTION, DERReplayBuffer, ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, PrioritizedReplayBufferSamples, ReplayBufferSamples, Schedule
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.dqn.policies import DQNPolicy

SelfDQNDER = TypeVar("SelfDQNDER", bound="DQNDER")


class DQNDER(DQN):
    """
    Deep Q-Network (Vanilla DQN 2015, with target network) with the double experience replay memory (DER):
    Han et al. (2021), Reinforcement Learning Guided by Double Replay Memory, Journal of Sensors,
    https://doi.org/10.1155/2021/6652042 (Algorithm 1, with the uniform + TD-error based sampling of Sec. 3.1).

    The memory budget ``buffer_size`` is split between two memories (see ``DERReplayBuffer``), in the proportions
    of the paper (N1 = 50,000, N2 = 12,800, ``der_h2_fraction``):
    - H1: every new transition, sampled uniformly
    - H2: copies of the H1 transitions used in an update, sampled with probability proportional to
      p = exp(|delta|) at insertion, then p <- p ** w at every reuse, w = delta^2 / sum_batch delta^2

    Training step:
        batch = round(der_lambda * B) samples from H2 (once it holds ``batch_size`` transitions) + the rest from H1
        y_j = r_j + gamma * (1 - done_j) * max_a Q_target(s'_j, a)
        loss = mean_j (Q(s_j, a_j) - y_j)^2 + tf_lambda * mean_j (Q(s_j, a_j) - Q_old(s_j, a_j))^2
            (SUFT term on the samples of both memories; no importance-sampling weights, as in the paper)
        then: update the priorities of the H2 samples (w with the TD errors of the whole batch),
              copy the H1 samples into H2 with p = exp(|delta|) (TD errors before the gradient step)
    Q_old(s, a) is the Q-value of the network when the transition was collected (it moves with the transition to H2).

    The other hyperparameters are the ones of the ``DQN`` agent of this repo (batch size 32, Adam with learning rate
    1e-4, target network updated every 10K steps, ...), not the ones of the paper (batch 128, learning rate 5e-4).

    :param der_lambda: Fraction of each batch taken from H2 (lambda in the paper; 0.1 is its best setting on Atari)
    :param der_h2_fraction: Fraction of the memory budget ``buffer_size`` given to H2
    See ``DQN`` for the other parameters (``buffer_size`` is the memory budget of H1 + H2).
    """

    # Replay buffer of the agent (the short / long term agent uses its short term + long term DER version)
    replay_buffer_cls: ClassVar[type[ReplayBuffer]] = DERReplayBuffer
    replay_buffer: DERReplayBuffer

    def __init__(
        self,
        policy: Union[str, type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # memory budget of H1 + H2
        der_lambda: float = 0.1,
        der_h2_fraction: float = DER_H2_FRACTION,
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
        replay_buffer_kwargs = {
            "der_lambda": der_lambda,
            "h2_fraction": der_h2_fraction,
            # H2 is sampled once it holds a full batch
            "h2_learning_starts": batch_size,
            **(replay_buffer_kwargs or {}),
        }
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

    def _per_sample_losses(
        self, replay_data: Union[ReplayBufferSamples, PrioritizedReplayBufferSamples]
    ) -> tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Per-sample TD error (detached), squared TD error and SUFT term ((Q - Q_old)^2, ``None`` if ``tf_lambda == 0``),
        all of shape [batch, 1].
        """
        with th.no_grad():
            # Compute the next Q-values using the target network, greedy policy
            next_q_values, _ = self.q_net_target(replay_data.next_observations).max(dim=1)
            # 1-step TD target
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values.reshape(-1, 1)

        # Get current Q-values estimates for the actions from the replay buffer
        current_q_values = th.gather(self.q_net(replay_data.observations), dim=1, index=replay_data.actions.long())
        td_errors = current_q_values - target_q_values
        # SUFT CHANGE
        suft_term = None
        if self.tf_lambda > 0:
            old_q_values = th.gather(replay_data.old_values, dim=1, index=replay_data.actions.long())
            suft_term = (current_q_values - old_q_values).pow(2)
        # SUFT CHANGE until here
        return td_errors.detach(), td_errors.pow(2), suft_term

    @staticmethod
    def _update_der(
        der_buffer: DERReplayBuffer,
        h1_data: Optional[PrioritizedReplayBufferSamples],
        h1_td: Optional[th.Tensor],
        h2_data: Optional[PrioritizedReplayBufferSamples],
        h2_td: Optional[th.Tensor],
    ) -> None:
        """
        DER memory updates after a batch: H2 priorities p <- p ** w, w = delta^2 / sum_n delta_n^2 over the DER samples
        of the batch (H1 + H2), then the used H1 transitions are copied into H2 with p = exp(|delta|).
        """
        tds = [td for td in (h1_td, h2_td) if td is not None]
        if not tds:
            return
        sum_sq = float(th.cat(tds).pow(2).sum())
        if h2_data is not None and h2_td is not None:
            weights = (h2_td.pow(2) / sum_sq).cpu().numpy() if sum_sq > 0 else np.zeros(len(h2_td))
            der_buffer.secondary.update_priorities(h2_data.indices, weights)
        if h1_data is not None and h1_td is not None:
            der_buffer.store_used(h1_data.indices, h1_td.cpu().numpy())

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses, h2_fractions = [], []
        for _ in range(gradient_steps):
            # DER batch: uniform H1 samples + TD-error based H2 samples
            h1_data, h2_data = self.replay_buffer.sample_der(batch_size, env=self._vec_normalize_env)
            loss = th.zeros((), device=self.device)
            td_by_memory: dict[str, Optional[th.Tensor]] = {"h1": None, "h2": None}
            for name, data in (("h1", h1_data), ("h2", h2_data)):
                if data is None:
                    continue
                td, td_loss, suft_term = self._per_sample_losses(data)
                td_by_memory[name] = td
                # SUFT CHANGE: the SUFT term on the samples of both memories
                loss = loss + td_loss.sum()
                if suft_term is not None:
                    loss = loss + suft_term.sum() * self.tf_lambda
            loss = loss / batch_size
            losses.append(loss.item())
            h2_fractions.append(0.0 if h2_data is None else len(h2_data.indices) / batch_size)

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            # DER: update the H2 priorities and copy the used H1 transitions into H2
            self._update_der(self.replay_buffer, h1_data, td_by_memory["h1"], h2_data, td_by_memory["h2"])

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/der_h2_batch_fraction", np.mean(h2_fractions))
        self.logger.record("train/der_h2_size", self.replay_buffer.secondary.size())

    def learn(
        self: SelfDQNDER,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DQNDER",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDQNDER:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
