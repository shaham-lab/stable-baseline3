from typing import Any, ClassVar, Optional, TypeVar, Union

import torch as th

from stable_baselines3.common.buffers import EREReplayBuffer, ReplayBuffer, ShortUniformLongEREReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.dqn.dqn_ere import DQNERE
from stable_baselines3.dqn.policies import DQNPolicy

SelfDQNSLERE = TypeVar("SelfDQNSLERE", bound="DQNSLERE")


class DQNSLERE(DQNERE):
    """
    Short-long term SUFT DQN (Vanilla DQN 2015, with target network) with Emphasizing Recent Experience (ERE,
    Wang et al. (2020), see ``DQNERE``) on the long term buffer only (see ``ShortUniformLongEREReplayBuffer``):

    - short term buffer (``buffer_size``): the newest transitions, always sampled uniformly
    - long term buffer (``long_term_buffer_size``): the transitions that left the short term buffer; in the ERE
      updates it is sampled from its c_k most recent transitions, c_k = max(N_long * eta^(k * 1000 / K), c_min)

    Every minibatch takes ``sampling_fraction`` of its samples from the short term buffer (all of them while the long
    term buffer has too few transitions). The ERE phases and the optional continuous DQN updates
    (``ere_continuous_updates``, uniform sampling in both buffers) are the ones of ``DQNERE``.
    As in ``DQNSL``, the SUFT term is added for the short term samples only:

        loss = 1/B [ sum_short ((Q - y)^2 + tf_lambda * (Q - Q_old)^2) + sum_long (Q - y)^2 ]

    :param long_term_buffer_size: size of the long term replay buffer
    :param sampling_fraction: fraction of each minibatch taken from the short term replay buffer
    See ``DQNERE`` / ``DQN`` for the other parameters (``buffer_size`` is the size of the short term buffer).
    """

    replay_buffer_cls: ClassVar[type[ReplayBuffer]] = ShortUniformLongEREReplayBuffer
    replay_buffer: ShortUniformLongEREReplayBuffer  # type: ignore[assignment]

    def __init__(
        self,
        policy: Union[str, type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 4000,  # short buffer 4K
        long_term_buffer_size: int = 1_000_000,  # 1e6
        sampling_fraction: float = 0.5,
        ere_eta0: float = 0.995,
        ere_c_min: int = 5000,
        ere_updates_per_step: float = 0.25,
        ere_continuous_updates: bool = False,
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
            ere_eta0=ere_eta0,
            ere_c_min=ere_c_min,
            ere_updates_per_step=ere_updates_per_step,
            ere_continuous_updates=ere_continuous_updates,
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

    def _ere_buffer(self) -> EREReplayBuffer:
        # ERE on the long term buffer only
        return self.replay_buffer.long_term_buffer

    def _sample_uniform(self, batch_size: int) -> Any:
        return self.replay_buffer.sample_short_and_long(batch_size, env=self._vec_normalize_env)

    def _sample_recent(self, batch_size: int, c_k: int) -> Any:
        return self.replay_buffer.sample_short_and_long_recent(batch_size, c_k, env=self._vec_normalize_env)

    def _loss(self, data: Any, batch_size: int) -> th.Tensor:
        short_data, long_data = data
        loss = th.zeros((), device=self.device)
        # SUFT CHANGE: the SUFT term on the short term samples only
        if short_data is not None:
            loss = loss + self._dqn_loss(short_data, suft=True, reduction="sum")
        if long_data is not None:
            loss = loss + self._dqn_loss(long_data, suft=False, reduction="sum")
        return loss / batch_size

    def learn(
        self: SelfDQNSLERE,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DQNSLERE",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDQNSLERE:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
