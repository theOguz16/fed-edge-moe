from dataclasses import dataclass

import torch


@dataclass
class ExpertJob:
    expert_id: int
    token_indices: torch.Tensor
    expert_input: torch.Tensor
    weights: torch.Tensor
    output_shape: tuple


def prepare_expert_job(
    x,
    route,
    expert_id,
):
    batch_size, seq_len, d_model = x.shape

    flat_x = x.reshape(
        -1,
        d_model,
    )

    flat_indices = (
        route.topk_indices.reshape(
            -1,
            route.topk_indices.shape[-1],
        )
    )

    flat_weights = (
        route.topk_weights.reshape(
            -1,
            route.topk_weights.shape[-1],
        )
    )

    selected = (
        flat_indices == expert_id
    )

    token_indices, slot_indices = (
        selected.nonzero(
            as_tuple=True
        )
    )

    expert_input = (
        flat_x[token_indices]
    )

    weights = (
        flat_weights[
            token_indices,
            slot_indices,
        ]
        .unsqueeze(-1)
    )

    return ExpertJob(
        expert_id=expert_id,
        token_indices=token_indices,
        expert_input=expert_input,
        weights=weights,
        output_shape=(
            batch_size,
            seq_len,
            d_model,
        ),
    )


def run_expert_job(
    expert,
    job,
):
    return expert(
        job.expert_input
    )


def merge_expert_result(
    flat_output,
    job,
    expert_output,
):
    weighted = (
        expert_output
        * job.weights
    )

    return flat_output.index_add(
        0,
        job.token_indices,
        weighted,
    )
