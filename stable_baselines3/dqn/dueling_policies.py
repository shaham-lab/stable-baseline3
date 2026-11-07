"""
Dueling network architecture for the DQN agents of this repo.

Paper: Wang et al. (2016), Dueling Network Architectures for Deep Reinforcement Learning,
https://arxiv.org/abs/1511.06581

The Q-network is split after the convolutional layers into two streams that share them:

    features  = conv(s)                                       (the 3 conv layers of the Nature DQN, flattened)
    V(s)      = value stream(features)                        (FC 512 -> 1)
    A(s, a)   = advantage stream(features)                    (FC 512 -> |A|)
    Q(s, a)   = V(s) + A(s, a) - 1/|A| sum_a' A(s, a')        (eq. 9 of the paper, mean aggregation)

As in the paper, the gradient that enters the last convolutional layer (the sum of the gradients of both streams)
is rescaled by 1/sqrt(2) (``rescale_gradient=True``, default).

The networks are drop-in replacements of ``QNetwork`` / ``DQNPolicy``: ``policy.q_net(obs)`` and
``policy.q_net_target(obs)`` return the Q-values of all the actions, so the agents (TD target with the target network,
SUFT term on the stored old Q-values, short / long term buffers) are unchanged.
"""

import math
from typing import Any, Optional

import gymnasium as gym
import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork


class NatureCNNConv(NatureCNN):
    """
    The convolutional layers of the Nature DQN CNN, without its fully connected layer:
    the features are the flattened output of the last convolutional layer (3136 for 84x84 frames).
    The fully connected layers (512 units) are in the value and advantage streams of the dueling network.

    :param observation_space: The observation space of the environment
    :param normalized_image: Whether to assume that the image is already normalized (see ``NatureCNN``)
    """

    def __init__(self, observation_space: gym.Space, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim=512, normalized_image=normalized_image)
        # Drop the fully connected layer of NatureCNN: the features are the flattened conv output
        self._features_dim = self.linear[0].in_features
        self.linear = nn.Identity()


class _ScaleGradient(th.autograd.Function):
    """Identity in the forward pass, multiplies the gradient by ``scale`` in the backward pass."""

    @staticmethod
    def forward(ctx, x: th.Tensor, scale: float) -> th.Tensor:  # type: ignore[override]
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: th.Tensor):  # type: ignore[override]
        return grad_output * ctx.scale, None


class DuelingQNetwork(QNetwork):
    """
    Dueling action-value (Q-value) network: a value stream and an advantage stream on shared features,
    Q(s, a) = V(s) + A(s, a) - mean_a' A(s, a').

    :param observation_space: Observation space
    :param action_space: Action space
    :param features_extractor: Shared features extractor (the conv layers for images)
    :param features_dim: Number of features
    :param net_arch: Hidden layers of each stream (the value and the advantage streams have the same architecture)
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not, dividing by 255.0 (True by default)
    :param rescale_gradient: Rescale the gradient entering the shared features by 1/sqrt(2) (as in the paper)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: Optional[list[int]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        rescale_gradient: bool = True,
    ) -> None:
        # QNetwork.__init__ would build a single Q head: only the BasePolicy part is used
        BasePolicy.__init__(
            self,
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )
        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_dim = features_dim
        self.rescale_gradient = rescale_gradient
        action_dim = int(self.action_space.n)  # number of actions
        self.value_stream = nn.Sequential(*create_mlp(self.features_dim, 1, self.net_arch, self.activation_fn))
        self.advantage_stream = nn.Sequential(*create_mlp(self.features_dim, action_dim, self.net_arch, self.activation_fn))

    def forward(self, obs: PyTorchObs) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        features = self.extract_features(obs, self.features_extractor)
        if self.rescale_gradient and features.requires_grad:
            # Both streams backpropagate into the shared layers: rescale their summed gradient by 1/sqrt(2)
            features = _ScaleGradient.apply(features, 1.0 / math.sqrt(2.0))
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Mean aggregation (eq. 9): the advantages have zero mean over the actions
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(dict(rescale_gradient=self.rescale_gradient))
        return data


class DuelingDQNPolicy(DQNPolicy):
    """
    Policy class with a dueling Q-value net and a dueling target net, for DQN.
    Same as ``DQNPolicy`` with ``DuelingQNetwork`` networks.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: Hidden layers of each stream (value and advantage)
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use (shared by both streams).
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param rescale_gradient: Rescale the gradient entering the shared features by 1/sqrt(2) (as in the paper)
    """

    q_net: DuelingQNetwork
    q_net_target: DuelingQNetwork

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[list[int]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        rescale_gradient: bool = True,
    ) -> None:
        # Needed by make_q_net(), which DQNPolicy.__init__ calls (through _build())
        self.rescale_gradient = rescale_gradient
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

    def make_q_net(self) -> DuelingQNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return DuelingQNetwork(**net_args, rescale_gradient=self.rescale_gradient).to(self.device)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(dict(rescale_gradient=self.rescale_gradient))
        return data


DuelingMlpPolicy = DuelingDQNPolicy


class DuelingCnnPolicy(DuelingDQNPolicy):
    """
    Dueling policy class for DQN when using images as input, as in the paper:
    the 3 conv layers of the Nature DQN are shared, then each stream has a fully connected layer of 512 units.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: Hidden layers of each stream (default ``[512]``, as in the paper)
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use (default: the conv layers of the Nature DQN).
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param rescale_gradient: Rescale the gradient entering the shared features by 1/sqrt(2) (as in the paper)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[list[int]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = NatureCNNConv,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        rescale_gradient: bool = True,
    ) -> None:
        if net_arch is None:
            # Paper: one fully connected layer of 512 units in each stream
            net_arch = [512]
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            rescale_gradient,
        )


class DuelingMultiInputPolicy(DuelingDQNPolicy):
    """
    Dueling policy class for DQN when using dict observations as input.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: Hidden layers of each stream (value and advantage)
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param rescale_gradient: Rescale the gradient entering the shared features by 1/sqrt(2) (as in the paper)
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[list[int]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        features_extractor_class: type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        rescale_gradient: bool = True,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            rescale_gradient,
        )
