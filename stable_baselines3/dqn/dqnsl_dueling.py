from typing import ClassVar, TypeVar

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.dqn.dqn_sl import DQNSL
from stable_baselines3.dqn.dueling_policies import (
    DuelingCnnPolicy,
    DuelingDQNPolicy,
    DuelingMlpPolicy,
    DuelingMultiInputPolicy,
    DuelingQNetwork,
)

SelfDQNSLDueling = TypeVar("SelfDQNSLDueling", bound="DQNSLDueling")


class DQNSLDueling(DQNSL):
    """
    Short-long term SUFT DQN (Vanilla DQN 2015, with target network) with the dueling network architecture
    (see ``dueling_policies.py``): Q(s, a) = V(s) + A(s, a) - mean_a' A(s, a').

    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Dueling: Wang et al. (2016), Dueling Network Architectures for Deep Reinforcement Learning,
    https://arxiv.org/abs/1511.06581

    Everything but the network is the ``DQNSL`` agent of this repo (same arguments and default hyperparameters):
    a short term buffer (``buffer_size``) and a long term buffer (``long_term_buffer_size``), every minibatch takes
    ``sampling_fraction`` of its samples from the short term buffer, the TD loss uses the target network,
    and the SUFT term (``tf_lambda > 0``) is added for the short term samples only:
        loss = 1/B [ sum_short ((Q - y)^2 + tf_lambda * (Q - Q_old)^2) + sum_long (Q - y)^2 ]

    The policies (``"MlpPolicy"``, ``"CnnPolicy"``, ``"MultiInputPolicy"``) are the dueling ones. The CNN policy has
    the 3 conv layers of the Nature DQN and one fully connected layer of 512 units per stream, as in the paper.
    Pass ``policy_kwargs=dict(rescale_gradient=False)`` to disable the 1/sqrt(2) rescaling of the gradient
    that enters the shared layers.

    See ``DQNSL`` for the parameters.
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
        self: SelfDQNSLDueling,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DQNSLDueling",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDQNSLDueling:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
