from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th

from stable_baselines3.common.buffers import NStepShortLongTermPrioritizedReplayBuffer, ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, PrioritizedReplayBufferSamples, Schedule
from stable_baselines3.dqn.ddqn_n_step_per_dueling import DDQNNStepPERDueling
from stable_baselines3.dqn.dueling_policies import DuelingDQNPolicy

SelfDDQNSLNStepPERDueling = TypeVar("SelfDDQNSLNStepPERDueling", bound="DDQNSLNStepPERDueling")


class DDQNSLNStepPERDueling(DDQNNStepPERDueling):
    """
    Short-long term SUFT version of ``DDQNNStepPERDueling`` (Double DQN target, n-step returns, PER with alpha = 0.5,
    dueling network, on the DQN with target network), with the short and long term buffers of ``DQNSL``
    (see ``NStepShortLongTermPrioritizedReplayBuffer``).

    Every new transition is added to the short term buffer (``buffer_size``); once it is full, the n-step transition
    of its oldest sample moves to the long term buffer (``long_term_buffer_size``). Every minibatch takes
    ``sampling_fraction`` of its samples from the short term buffer (all of them while the long term buffer has
    too few samples). The long term buffer is prioritized; the short term one is prioritized too unless
    ``prioritize_short_term=False`` (then it is sampled uniformly, as ``DQNSLPERLong``).
    The PER loss weights are normalized over each part of the batch, and the SUFT term is added for the short term
    samples only (as in ``DQNSLPERBoth``):

        loss = 1/B [ sum_short w_j ((Q_j - y_j)^2 + tf_lambda * (Q_j - Q_old_j)^2) + sum_long w_j (Q_j - y_j)^2 ]

    with y_j the n-step Double DQN target (see ``DDQNNStepPERDueling``) and w_j = 1 for uniform short term samples.

    :param long_term_buffer_size: size of the long term replay buffer
    :param sampling_fraction: fraction of each minibatch taken from the short term replay buffer
    :param prioritize_short_term: whether the short term buffer is prioritized (otherwise sampled uniformly)
    See ``DDQNNStepPERDueling`` for the other parameters (``buffer_size`` is the size of the short term buffer).
    """

    replay_buffer_cls: ClassVar[type[ReplayBuffer]] = NStepShortLongTermPrioritizedReplayBuffer
    replay_buffer: NStepShortLongTermPrioritizedReplayBuffer

    def __init__(
        self,
        policy: Union[str, type[DuelingDQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 4000,  # short buffer 4K
        long_term_buffer_size: int = 1_000_000,  # 1e6
        sampling_fraction: float = 0.5,
        prioritize_short_term: bool = True,
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        n_steps: int = 3,  # Rainbow
        per_priority_exponent: float = 0.5,  # Rainbow
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
        # Short term and long term replay buffers, `buffer_size` is the size of the short term one
        replay_buffer_kwargs = {
            **(replay_buffer_kwargs or {}),
            "long_term_buffer_size": long_term_buffer_size,
            "sampling_fraction": sampling_fraction,
            "prioritize_short_term": prioritize_short_term,
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
            replay_buffer_kwargs=replay_buffer_kwargs,
            n_steps=n_steps,
            per_priority_exponent=per_priority_exponent,
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

        losses = []
        self._per_mean_weights: dict[str, list[float]] = {"short": [], "long": []}
        for _ in range(gradient_steps):
            # Sample replay buffer (n-step transitions, prioritized long term part)
            short_replay_data, long_replay_data = self.replay_buffer.sample_short_and_long(
                batch_size, env=self._vec_normalize_env
            )
            short_replay_loss_sum = self.calculate_loss(short_replay_data, True)
            long_replay_loss_sum = self.calculate_loss(long_replay_data, False)
            loss = (short_replay_loss_sum + long_replay_loss_sum) / batch_size
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        for part, weights in self._per_mean_weights.items():
            if weights:
                self.logger.record(f"train/per_mean_weight_{part}", np.mean(weights))

    def calculate_loss(
        self, replay_data: Optional[PrioritizedReplayBufferSamples], is_short: bool
    ) -> Union[th.Tensor, float]:
        """
        Loss summed over the samples of one part of the batch (the caller divides by the batch size),
        and update of their priorities.

        :param replay_data: Samples from one of the two buffers (``None`` if nothing was sampled from it)
        :param is_short: Whether the samples come from the short term buffer,
            the SUFT term is only added for those
        :return: Sum over the samples of the PER weighted TD loss (plus the SUFT term scaled by ``tf_lambda``
            for the short term samples), ``0.0`` if nothing was sampled
        """
        if replay_data is None:
            return 0.0
        prioritized = not is_short or self.replay_buffer.prioritize_short_term
        if prioritized:
            # Weight the loss by the inverse priorities (normalized over this part of the batch)
            loss_weights = self._per_loss_weights(replay_data)
            self._per_mean_weights["short" if is_short else "long"].append(loss_weights.mean().item())
        else:
            # Uniform short term samples
            loss_weights = th.ones_like(replay_data.rewards)
        td_loss, suft_term = self._per_sample_losses(replay_data)

        if prioritized:
            # Update the priorities of the sampled transitions with their TD errors (not the SUFT term)
            buffer = self.replay_buffer if is_short else self.replay_buffer.long_term_buffer
            buffer.update_priorities(replay_data.indices, self._priorities(td_loss))

        # SUFT CHANGE
        loss = (loss_weights * td_loss).sum()
        # SUFT term, short term samples only (also weighted by the importance-sampling loss weights)
        if is_short and suft_term is not None:
            loss = loss + (loss_weights * suft_term).sum() * self.tf_lambda
        # SUFT CHANGE until here
        return loss

    def learn(
        self: SelfDDQNSLNStepPERDueling,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DDQNSLNStepPERDueling",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDDQNSLNStepPERDueling:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
