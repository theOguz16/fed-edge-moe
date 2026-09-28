import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def _cpu_clone_state_dict(state_dict):
    return {
        key: value.detach().cpu().clone().contiguous()
        for key, value in state_dict.items()
    }


def export_global_snapshot(
    model,
    output_dir,
    global_version=0,
):
    """
    Global modeli üç parçaya ayırır:

    1. shared parameters
    2. routers
    3. individual experts
    """

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    full_state = model.state_dict()

    # Expert ve router dışındaki her şey shared kabul edilir.
    shared_state = {
        key: value
        for key, value in full_state.items()
        if ".moe.experts." not in key
        and ".moe.router." not in key
    }

    save_file(
        _cpu_clone_state_dict(shared_state),
        str(root / "shared.safetensors"),
    )

    router_dir = root / "routers"
    router_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    expert_root = root / "experts"
    expert_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "format_version": 1,
        "global_version": global_version,
        "num_layers": model.num_layers,
        "num_experts": model.num_experts,
        "shared": "shared.safetensors",
        "routers": {},
        "experts": {},
    }

    # --------------------------------------------------------
    # Routers
    # --------------------------------------------------------

    for layer_idx, block in enumerate(model.blocks):

        router_file = (
            router_dir
            / f"layer_{layer_idx:02d}_router.safetensors"
        )

        save_file(
            _cpu_clone_state_dict(
                block.moe.router.state_dict()
            ),
            str(router_file),
        )

        manifest["routers"][str(layer_idx)] = {
            "version": global_version,
            "file": str(
                router_file.relative_to(root)
            ),
        }

    # --------------------------------------------------------
    # Experts
    # --------------------------------------------------------

    for layer_idx, block in enumerate(model.blocks):

        layer_dir = (
            expert_root
            / f"layer_{layer_idx:02d}"
        )

        layer_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        manifest["experts"][
            str(layer_idx)
        ] = {}

        for expert_idx, expert in enumerate(
            block.moe.experts
        ):

            expert_file = (
                layer_dir
                / (
                    f"expert_{expert_idx:02d}"
                    f"_v{global_version:04d}"
                    f".safetensors"
                )
            )

            save_file(
                _cpu_clone_state_dict(
                    expert.state_dict()
                ),
                str(expert_file),
            )

            manifest["experts"][
                str(layer_idx)
            ][str(expert_idx)] = {
                "version": global_version,
                "file": str(
                    expert_file.relative_to(root)
                ),
            }

    with open(
        root / "manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    return root


def load_global_snapshot(
    model,
    snapshot_dir,
):
    root = Path(snapshot_dir)

    with open(
        root / "manifest.json",
        "r",
        encoding="utf-8",
    ) as f:
        manifest = json.load(f)

    # --------------------------------------------------------
    # Shared
    # --------------------------------------------------------

    shared_state = load_file(
        str(root / manifest["shared"])
    )

    model.load_state_dict(
        shared_state,
        strict=False,
    )

    # --------------------------------------------------------
    # Routers
    # --------------------------------------------------------

    for layer_idx in range(
        manifest["num_layers"]
    ):
        router_info = manifest[
            "routers"
        ][str(layer_idx)]

        router_state = load_file(
            str(root / router_info["file"])
        )

        model.blocks[
            layer_idx
        ].moe.router.load_state_dict(
            router_state
        )

    # --------------------------------------------------------
    # Experts
    # --------------------------------------------------------

    for layer_idx in range(
        manifest["num_layers"]
    ):
        for expert_idx in range(
            manifest["num_experts"]
        ):
            expert_info = manifest[
                "experts"
            ][str(layer_idx)][
                str(expert_idx)
            ]

            expert_state = load_file(
                str(
                    root
                    / expert_info["file"]
                )
            )

            model.blocks[
                layer_idx
            ].moe.experts[
                expert_idx
            ].load_state_dict(
                expert_state
            )

    return manifest


def export_single_expert(
    model,
    layer_idx,
    expert_idx,
    output_file,
):
    expert = (
        model
        .blocks[layer_idx]
        .moe
        .experts[expert_idx]
    )

    output_file = Path(output_file)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_file(
        _cpu_clone_state_dict(
            expert.state_dict()
        ),
        str(output_file),
    )


def load_single_expert(
    model,
    layer_idx,
    expert_idx,
    expert_file,
):
    state = load_file(
        str(expert_file)
    )

    expert = (
        model
        .blocks[layer_idx]
        .moe
        .experts[expert_idx]
    )

    expert.load_state_dict(state)
