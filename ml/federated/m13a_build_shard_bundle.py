import json
import shutil
import tarfile
from pathlib import Path


SOURCE = Path(
    "checkpoints/m10_candidates/global_step_0100"
)

OUTPUT_ROOT = Path(
    "checkpoints/m13a"
)

TARGET_LAYER = 0
TARGET_EXPERT = 7
GLOBAL_VERSION = 100


def main():
    bundle = (
        OUTPUT_ROOT
        / f"v{GLOBAL_VERSION:04d}_L{TARGET_LAYER}_E{TARGET_EXPERT}"
    )

    if bundle.exists():
        shutil.rmtree(bundle)

    bundle.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        SOURCE / "shared.safetensors",
        bundle / "shared.safetensors",
    )

    routers_dst = bundle / "routers"

    shutil.copytree(
        SOURCE / "routers",
        routers_dst,
    )

    expert_src = (
        SOURCE
        / "experts"
        / f"layer_{TARGET_LAYER:02d}"
        / (
            f"expert_{TARGET_EXPERT:02d}"
            f"_v{GLOBAL_VERSION:04d}.safetensors"
        )
    )

    expert_dst_dir = (
        bundle
        / "experts"
        / f"layer_{TARGET_LAYER:02d}"
    )

    expert_dst_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    expert_dst = (
        expert_dst_dir
        / expert_src.name
    )

    shutil.copy2(
        expert_src,
        expert_dst,
    )

    manifest = {
        "global_version":
            GLOBAL_VERSION,
        "target_layer":
            TARGET_LAYER,
        "target_expert":
            TARGET_EXPERT,
        "bundle_type":
            "expert_training_shard",
        "files": {
            "shared":
                "shared.safetensors",
            "routers":
                "routers/",
            "expert":
                str(
                    expert_dst.relative_to(
                        bundle
                    )
                ),
        },
    }

    with open(
        bundle / "manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    tar_path = Path(
        str(bundle) + ".tar.gz"
    )

    if tar_path.exists():
        tar_path.unlink()

    with tarfile.open(
        tar_path,
        "w:gz",
    ) as tar:
        tar.add(
            bundle,
            arcname="shard_bundle",
        )

    full_bytes = sum(
        p.stat().st_size
        for p in SOURCE.rglob("*")
        if p.is_file()
    )

    bundle_bytes = sum(
        p.stat().st_size
        for p in bundle.rglob("*")
        if p.is_file()
    )

    tar_bytes = (
        tar_path.stat().st_size
    )

    print()
    print("FedEdgeMoE - M13A")
    print("Expert Shard Delivery Bundle")
    print()

    print(
        "Target:",
        f"L{TARGET_LAYER}-E{TARGET_EXPERT}",
    )

    print(
        "Full snapshot:",
        f"{full_bytes / 1_000_000:.3f} MB",
    )

    print(
        "Raw shard bundle:",
        f"{bundle_bytes / 1_000_000:.3f} MB",
    )

    print(
        "Compressed bundle:",
        f"{tar_bytes / 1_000_000:.3f} MB",
    )

    print(
        "Download reduction:",
        f"{(1 - tar_bytes / full_bytes) * 100:.1f}%",
    )

    print(
        "Compression ratio:",
        f"{full_bytes / tar_bytes:.2f}x",
    )

    print()
    print(
        "Bundle:",
        bundle,
    )

    print(
        "Archive:",
        tar_path,
    )


if __name__ == "__main__":
    main()
