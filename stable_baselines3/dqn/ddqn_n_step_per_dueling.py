from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th

from stable_baselines3.common.buffers import NStepPrioritizedReplayBuffer, ReplayBuffer
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, PrioritizedReplayBufferSamples, Schedule
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.dqn.dueling_policies import (
    DuelingCnnPolicy,
    DuelingDQNPolicy,
    DuelingMlpPolicy,
    DuelingMultiInputPolicy,
    DuelingQNetwork,
)

SelfDDQNNStepPERDueling = TypeVar("SelfDDQNNStepPERDueling", bound="DDQNNStepPERDueling")


class DDQNNStepPERDueling(DQN):
    """
    DQN (Vanilla DQN 2015, with target network) combining four Rainbow components
    (Hessel et al., 2018, https://arxiv.org/abs/1710.02298), without the distributional loss and the noisy nets:

    - Double DQN target (van Hasselt et al., 2016, https://arxiv.org/abs/1509.06461):
      the online network selects the next action, the target network evaluates it
    - n-step returns (Rainbow ``n_steps=3``)
    - Prioritized Experience Replay, proportional variant (Schaul et al., 2016, https://arxiv.org/abs/1511.05952),
      priority exponent alpha = 0.5 (``per_priority_exponent``, as Rainbow / Dopamine)
    - Dueling network architecture (Wang et al., 2016, https://arxiv.org/abs/1511.06581), see ``dueling_policies.py``

    Training step:
        sample j ~ P(j) = p_j^alpha / sum_k p_k^alpha (stratified), new transitions get the max recorded priority
        loss_weights_j = 1 / sqrt(P(j) + 1e-10), normalized by their max over the batch
            (fixed importance-sampling exponent beta = 0.5, as the ``DQNPER`` of this repo / Dopamine)
        a*_j = argmax_a Q(s_{j+k}, a)                                  (online network)
        y_j = R_j^(n) + gamma^k * (1 - done_{j+k}) * Q_target(s_{j+k}, a*_j)
            R_j^(n) = sum_{i<k} gamma^i r_{j+i+1}, k = n_steps, or the number of steps until the episode ends
        loss = mean_j loss_weights_j * (Q(s_j, a_j) - y_j)^2   (MSE, as the DQN of this repo)
             + tf_lambda * mean_j loss_weights_j * (Q(s_j, a_j) - Q_old(s_j, a_j))^2   (SUFT term, tf_lambda > 0)
        p_j <- |Q(s_j, a_j) - y_j|  (the TD error only, not the SUFT term), raised to alpha by the buffer
    Q_old(s, a) is the Q-value of the (dueling) network when the transition was stored.

    The other hyperparameters are the ones of the ``DQN`` agent of this repo (target network updated every
    10K steps, Adam with learning rate 1e-4, gradient norm clipped at 10, ...).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...), the dueling ones
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
    :param replay_buffer_kwargs: Additional keyword arguments to pass to the replay buffer on creation.
    :param n_steps: Number of steps of the n-step return (Rainbow uses 3). ``n_steps=1`` is the 1-step update.
    :param per_priority_exponent: PER priority exponent alpha, applied to |TD error| (Rainbow: 0.5)
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
        (e.g. ``dict(rescale_gradient=False)``, see ``DuelingDQNPolicy``)
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param tf_lambda: Weight of the SUFT term (0: no SUFT term)
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": DuelingMlpPolicy,
        "CnnPolicy": DuelingCnnPolicy,
        "MultiInputPolicy": DuelingMultiInputPolicy,
    }
    # Replay buffer of the agent (the short / long term agent uses its short / long term version)
    replay_buffer_cls: ClassVar[type[ReplayBuffer]] = NStepPrioritizedReplayBuffer
    q_net: DuelingQNetwork
    q_net_target: DuelingQNetwork
    policy: DuelingDQNPolicy
    replay_buffer: NStepPrioritizedReplayBuffer

    def __init__(
        self,
        policy: Union[str, type[DuelingDQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
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
        # n-step returns and PER are both handled by the replay buffer
        replay_buffer_kwargs = {
            **(replay_buffer_kwargs or {}),
            "priority_exponent": per_priority_exponent,
            "n_steps": n_steps,
            "gamma": gamma,
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

    @staticmethod
    def _per_loss_weights(replay_data: PrioritizedReplayBufferSamples) -> th.Tensor:
        """PER importance-sampling loss weights (Dopamine): 1 / sqrt(P(j)), normalized by their max."""
        loss_weights = 1.0 / th.sqrt(replay_data.sampling_probabilities + 1e-10)
        return loss_weights / loss_weights.max()

    def _per_sample_losses(self, replay_data: PrioritizedReplayBufferSamples) -> tuple[th.Tensor, Optional[th.Tensor]]:
        """
        Per-sample squared n-step Double DQN TD error and SUFT term ((Q - Q_old)^2, ``None`` if ``tf_lambda == 0``),
        both of shape [batch, 1].
        """
        with th.no_grad():
            # Double DQN: the online network selects the next action, the target network evaluates it
            next_actions = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)
            next_q_values = th.gather(self.q_net_target(replay_data.next_observations), dim=1, index=next_actions)
            # n-step TD target (rewards hold the discounted n-step return, discounts is gamma ** k)
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * replay_data.discounts * next_q_values

        # Get current Q-values estimates for the actions from the replay buffer
        current_q_values = th.gather(self.q_net(replay_data.observations), dim=1, index=replay_data.actions.long())
        td_loss = (current_q_values - target_q_values).pow(2)
        # SUFT CHANGE
        suft_term = None
        if self.tf_lambda > 0:
            old_q_values = th.gather(replay_data.old_values, dim=1, index=replay_data.actions.long())
            suft_term = (current_q_values - old_q_values).pow(2)
        # SUFT CHANGE until here
        return td_loss, suft_term

    @staticmethod
    def _priorities(td_loss: th.Tensor) -> np.ndarray:
        """New priorities: |TD error| = sqrt(squared TD error + 1e-10) (the 1e-10 avoids zero priorities)."""
        return th.sqrt(td_loss.detach() + 1e-10).cpu().numpy()

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses, mean_weights = [], []
        for _ in range(gradient_steps):
            # Sample replay buffer (prioritized, n-step transitions), with the sampling probabilities
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            loss_weights = self._per_loss_weights(replay_data)
            td_loss, suft_term = self._per_sample_losses(replay_data)

            # MSE loss weighted by the importance-sampling loss weights (PER)
            loss = (loss_weights * td_loss).mean()
            # SUFT term (also weighted by the importance-sampling loss weights)
            if suft_term is not None:
                loss = loss + (loss_weights * suft_term).mean() * self.tf_lambda
            losses.append(loss.item())
            mean_weights.append(loss_weights.mean().item())

            # Update the priorities of the sampled transitions with their TD errors (not the SUFT term)
            self.replay_buffer.update_priorities(replay_data.indices, self._priorities(td_loss))

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
        self.logger.record("train/per_mean_weight", np.mean(mean_weights))

    def learn(
        self: SelfDDQNNStepPERDueling,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DDQNNStepPERDueling",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDDQNNStepPERDueling:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
