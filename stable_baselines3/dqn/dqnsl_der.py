from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th

from stable_baselines3.common.buffers import DER_H2_FRACTION, ReplayBuffer, ShortUniformLongDERReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.dqn.dqn_der import DQNDER
from stable_baselines3.dqn.policies import DQNPolicy

SelfDQNSLDER = TypeVar("SelfDQNSLDER", bound="DQNSLDER")


class DQNSLDER(DQNDER):
    """
    Short-long term SUFT DQN (Vanilla DQN 2015, with target network) whose long term buffer is a double experience
    replay memory (DER, Han et al. (2021), see ``DQNDER``), see ``ShortUniformLongDERReplayBuffer``:

    - short term buffer (``buffer_size``): the newest transitions, sampled uniformly
    - long term buffer (``long_term_buffer_size``, e.g. 1M - short term size): a DER memory, H1 (the transitions
      that left the short term buffer, uniform) + H2 (the used long term H1 transitions, TD-error priorities),
      split in the proportions of the paper (``der_h2_fraction``)

    Every minibatch takes ``sampling_fraction`` of its samples from the short term buffer (all of them while the long
    term H1 has too few transitions); the long term part is a DER batch (``der_lambda`` of it from H2).
    As in ``DQNSL``, the SUFT term is added for the short term samples only:

        loss = 1/B [ sum_short ((Q - y)^2 + tf_lambda * (Q - Q_old)^2) + sum_long (Q - y)^2 ]

    After the update, the H2 priorities are updated (w normalized over the long term samples of the batch) and the
    used long term H1 transitions are copied into H2. Batch size 32 and learning rate 1e-4 as the ``DQN`` of this repo.

    :param long_term_buffer_size: Memory budget of the long term DER (H1 + H2)
    :param sampling_fraction: Fraction of each minibatch taken from the short term buffer
    :param der_lambda: Fraction of the long term part of each batch taken from H2
    :param der_h2_fraction: Fraction of the long term memory budget given to H2
    See ``DQNDER`` / ``DQN`` for the other parameters (``buffer_size`` is the size of the short term buffer).
    """

    replay_buffer_cls: ClassVar[type[ReplayBuffer]] = ShortUniformLongDERReplayBuffer
    replay_buffer: ShortUniformLongDERReplayBuffer  # type: ignore[assignment]

    def __init__(
        self,
        policy: Union[str, type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 4000,  # short buffer 4K
        long_term_buffer_size: int = 996_000,  # long term DER budget: 1M - short buffer
        sampling_fraction: float = 0.5,
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
            "long_term_buffer_size": long_term_buffer_size,
            "sampling_fraction": sampling_fraction,
            **(replay_buffer_kwargs or {}),
        }
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            der_lambda=der_lambda,
            der_h2_fraction=der_h2_fraction,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
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

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses, h2_fractions = [], []
        for _ in range(gradient_steps):
            # Short term samples (uniform) + long term DER samples (H1 uniform, H2 TD-error based)
            short_data, long_h1, long_h2 = self.replay_buffer.sample_short_and_long_der(
                batch_size, env=self._vec_normalize_env
            )
            loss = th.zeros((), device=self.device)
            td_by_memory: dict[str, Optional[th.Tensor]] = {"h1": None, "h2": None}
            for name, data in (("short", short_data), ("h1", long_h1), ("h2", long_h2)):
                if data is None:
                    continue
                td, td_loss, suft_term = self._per_sample_losses(data)
                if name != "short":
                    td_by_memory[name] = td
                loss = loss + td_loss.sum()
                # SUFT CHANGE: the SUFT term on the short term samples only
                if name == "short" and suft_term is not None:
                    loss = loss + suft_term.sum() * self.tf_lambda
            loss = loss / batch_size
            losses.append(loss.item())
            h2_fractions.append(0.0 if long_h2 is None else len(long_h2.indices) / batch_size)

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            # DER (long term): update the H2 priorities and copy the used long term H1 transitions into H2
            self._update_der(self.replay_buffer.long_term_buffer, long_h1, td_by_memory["h1"], long_h2, td_by_memory["h2"])

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/der_h2_batch_fraction", np.mean(h2_fractions))
        self.logger.record("train/der_h2_size", self.replay_buffer.long_term_buffer.secondary.size())

    def learn(
        self: SelfDQNSLDER,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DQNSLDER",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDQNSLDER:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
