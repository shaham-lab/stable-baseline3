from typing import ClassVar, TypeVar

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.dqn.dueling_policies import (
    DuelingCnnPolicy,
    DuelingDQNPolicy,
    DuelingMlpPolicy,
    DuelingMultiInputPolicy,
    DuelingQNetwork,
)

SelfDQNDueling = TypeVar("SelfDQNDueling", bound="DQNDueling")


class DQNDueling(DQN):
    """
    Deep Q-Network (Vanilla DQN 2015, with target network) with the dueling network architecture
    (see ``dueling_policies.py``): Q(s, a) = V(s) + A(s, a) - mean_a' A(s, a').

    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Dueling: Wang et al. (2016), Dueling Network Architectures for Deep Reinforcement Learning,
    https://arxiv.org/abs/1511.06581

    Everything but the network is the ``DQN`` agent of this repo (same arguments and default hyperparameters):
    1-step TD target with the target network, MSE loss, gradient norm clipped at 10 (as in the paper),
    and the SUFT term when ``tf_lambda > 0``:
        loss = MSE(Q(s, a), y) + tf_lambda * MSE(Q(s, a), Q_old(s, a))
    where Q_old(s, a) is the Q-value of the (dueling) network when the transition was stored.
    Note: the paper's agent is the dueling Double DQN; this one keeps the DQN target to compare with ``DQN``.

    The policies (``"MlpPolicy"``, ``"CnnPolicy"``, ``"MultiInputPolicy"``) are the dueling ones. The CNN policy has
    the 3 conv layers of the Nature DQN and one fully connected layer of 512 units per stream, as in the paper.
    Pass ``policy_kwargs=dict(rescale_gradient=False)`` to disable the 1/sqrt(2) rescaling of the gradient
    that enters the shared layers.

    See ``DQN`` for the parameters.
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": DuelingMlpPolicy,
        "CnnPolicy": DuelingCnnPolicy,
        "MultiInputPolicy": DuelingMultiInputPolicy,
    }
    q_net: DuelingQNetwork
    q_net_target: DuelingQNetwork
    policy: DuelingDQNPolicy

    def learn(
        self: SelfDQNDueling,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DQNDueling",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDQNDueling:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
