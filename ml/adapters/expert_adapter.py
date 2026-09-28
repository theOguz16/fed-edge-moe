from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ExpertAdapter(ABC):
    """
    Generic interface between FedEdgeMoE
    and a concrete MoE expert implementation.
    """

    def __init__(self, expert: nn.Module):
        self.expert = expert

    @abstractmethod
    def projection_layers(self):
        """
        Return trainable projection layers.

        Expected format:
        {
            "gate_proj": nn.Linear,
            "up_proj": nn.Linear,
            "down_proj": nn.Linear,
        }
        """
        raise NotImplementedError

    def freeze(self):
        for p in self.expert.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.expert.parameters():
            p.requires_grad = True

    def parameter_count(self):
        return sum(
            p.numel()
            for p in self.expert.parameters()
        )

    def state_dict(self):
        return {
            k: v.detach().cpu().clone()
            for k, v
            in self.expert.state_dict().items()
        }

    def load_state_dict(self, state):
        self.expert.load_state_dict(state)

    def delta(
        self,
        before,
        after,
    ):
        return {
            key: after[key] - before[key]
            for key in before
        }

    def delta_l2_norm(self, delta):
        total = torch.tensor(0.0)

        for tensor in delta.values():
            total += tensor.float().pow(2).sum()

        return float(
            total.sqrt()
        )


class MiniMoEExpertAdapter(
    ExpertAdapter
):
    """
    Adapter for the current FedEdgeMoE ExpertMLP.
    """

    def projection_layers(self):
        required = [
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

        result = {}

        for name in required:
            layer = getattr(
                self.expert,
                name,
                None,
            )

            if not isinstance(
                layer,
                nn.Linear,
            ):
                raise TypeError(
                    f"{name} is not nn.Linear"
                )

            result[name] = layer

        return result


class MappedExpertAdapter(ExpertAdapter):
    """
    Generic adapter for experts whose projection
    layer names differ from FedEdgeMoE's native names.

    Example:
    {
        "gate_proj": "w1",
        "up_proj": "w3",
        "down_proj": "w2",
    }
    """

    def __init__(
        self,
        expert,
        projection_map,
    ):
        super().__init__(expert)

        self.projection_map = projection_map

    def projection_layers(self):
        result = {}

        for logical_name, actual_name in (
            self.projection_map.items()
        ):
            layer = getattr(
                self.expert,
                actual_name,
                None,
            )

            if not isinstance(
                layer,
                nn.Linear,
            ):
                raise TypeError(
                    f"{actual_name} is not nn.Linear"
                )

            result[logical_name] = layer

        return result


def _replace_projection_default(
    adapter,
    logical_name,
    new_layer,
):
    setattr(
        adapter.expert,
        logical_name,
        new_layer,
    )


def _replace_projection_mapped(
    adapter,
    logical_name,
    new_layer,
):
    actual_name = (
        adapter.projection_map[
            logical_name
        ]
    )

    setattr(
        adapter.expert,
        actual_name,
        new_layer,
    )


ExpertAdapter.replace_projection = (
    _replace_projection_default
)

MappedExpertAdapter.replace_projection = (
    _replace_projection_mapped
)
