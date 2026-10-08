from pathlib import Path
import argparse
import csv
import gc
import time

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    convnext_base,
    ConvNeXt_Base_Weights,
    convnext_large,
    ConvNeXt_Large_Weights,
)


ROOT = Path(
    "data/imagenetv2/"
    "imagenetv2-matched-frequency-format-val"
)


MODELS = {
    "resnet50": (
        resnet50,
        ResNet50_Weights.DEFAULT,
    ),
    "convnext_base": (
        convnext_base,
        ConvNeXt_Base_Weights.DEFAULT,
    ),
    "convnext_large": (
        convnext_large,
        ConvNeXt_Large_Weights.DEFAULT,
    ),
}



class ImageNetV2NumericLabels(Dataset):
    def __init__(self, root, transform):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        class_ids = sorted(
            int(p.name)
            for p in self.root.iterdir()
            if p.is_dir()
        )

        for class_id in class_ids:
            class_dir = self.root / str(class_id)

            images = sorted(
                list(class_dir.glob("*.jpeg"))
                + list(class_dir.glob("*.jpg"))
                + list(class_dir.glob("*.png"))
            )

            for path in images:
                self.samples.append(
                    (path, class_id)
                )

        if len(self.samples) != 10000:
            raise RuntimeError(
                f"Expected 10000 images, "
                f"found {len(self.samples)}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        with Image.open(path) as img:
            img = img.convert("RGB")
            img = self.transform(img)

        return img, label


def select_device(name):
    if name != "auto":
        return torch.device(name)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def run_precision(
    model_name,
    precision,
    dataset,
    device,
    batch_size,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model_fn, weights = MODELS[model_name]

    model = model_fn(
        weights=weights
    ).eval().to(device)

    if precision == "fp16":
        model = model.half()

    all_logits = []
    all_labels = []

    start = time.perf_counter()

    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)

            if precision == "fp16":
                images = images.half()
            else:
                images = images.float()

            logits = model(images)

            all_logits.append(
                logits.float().cpu()
            )
            all_labels.append(
                labels.cpu()
            )

            if (batch_idx + 1) % 100 == 0:
                done = min(
                    (batch_idx + 1) * batch_size,
                    len(dataset),
                )
                print(
                    f"{precision}: "
                    f"{done}/{len(dataset)}"
                )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    elapsed = time.perf_counter() - start

    logits = torch.cat(
        all_logits,
        dim=0,
    )
    labels = torch.cat(
        all_labels,
        dim=0,
    )

    del model
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    return logits, labels, elapsed


def topk_correct(
    logits,
    labels,
    k,
):
    indices = logits.topk(
        k,
        dim=1,
    ).indices

    return (
        indices == labels.unsqueeze(1)
    ).any(dim=1)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=sorted(MODELS),
        required=True,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--tag",
        default="mac",
    )

    args = parser.parse_args()

    device = select_device(
        args.device
    )

    print("Device:", device)
    print("Batch size:", args.batch_size)

    _, weights = MODELS[args.model]
    transform = weights.transforms()

    dataset = ImageNetV2NumericLabels(
        ROOT,
        transform,
    )

    print(
        "Dataset images:",
        len(dataset),
    )

    print("\nRunning FP32...")
    logits32, labels32, time32 = run_precision(
        args.model,
        "fp32",
        dataset,
        device,
        args.batch_size,
    )

    print("\nRunning FP16...")
    logits16, labels16, time16 = run_precision(
        args.model,
        "fp16",
        dataset,
        device,
        args.batch_size,
    )

    assert torch.equal(
        labels32,
        labels16,
    )

    labels = labels32

    correct1_32 = topk_correct(
        logits32,
        labels,
        1,
    )
    correct1_16 = topk_correct(
        logits16,
        labels,
        1,
    )

    correct5_32 = topk_correct(
        logits32,
        labels,
        5,
    )
    correct5_16 = topk_correct(
        logits16,
        labels,
        5,
    )

    pred32 = logits32.argmax(dim=1)
    pred16 = logits16.argmax(dim=1)

    top1_agree = (
        pred32 == pred16
    )

    top5_32 = logits32.topk(
        5,
        dim=1,
    ).indices

    top5_16 = logits16.topk(
        5,
        dim=1,
    ).indices

    top5_overlap = []

    for a, b in zip(
        top5_32,
        top5_16,
    ):
        overlap = len(
            set(a.tolist())
            & set(b.tolist())
        ) / 5.0

        top5_overlap.append(
            overlap
        )

    logp32 = F.log_softmax(
        logits32,
        dim=1,
    )

    logp16 = F.log_softmax(
        logits16,
        dim=1,
    )

    p32 = logp32.exp()

    per_image_kl = (
        p32
        * (
            logp32
            - logp16
        )
    ).sum(dim=1)

    prob32 = F.softmax(
        logits32,
        dim=1,
    )

    prob16 = F.softmax(
        logits16,
        dim=1,
    )

    conf32 = prob32.max(
        dim=1
    ).values

    conf16 = prob16.max(
        dim=1
    ).values

    conf_drift = (
        conf32
        - conf16
    ).abs()

    max_logit_diff = (
        logits32
        - logits16
    ).abs().max(
        dim=1
    ).values

    summary = {
        "model": args.model,
        "dataset": "imagenetv2_matched_frequency",
        "images": len(dataset),
        "device": str(device),
        "batch_size": args.batch_size,

        "fp32_top1_accuracy_pct":
            correct1_32.float().mean().item() * 100,

        "fp16_top1_accuracy_pct":
            correct1_16.float().mean().item() * 100,

        "fp32_top5_accuracy_pct":
            correct5_32.float().mean().item() * 100,

        "fp16_top5_accuracy_pct":
            correct5_16.float().mean().item() * 100,

        "top1_agreement_pct":
            top1_agree.float().mean().item() * 100,

        "mean_top5_overlap_pct":
            sum(top5_overlap)
            / len(top5_overlap)
            * 100,

        "mean_kl_fp32_to_fp16":
            per_image_kl.mean().item(),

        "mean_confidence_drift":
            conf_drift.mean().item(),

        "max_confidence_drift":
            conf_drift.max().item(),

        "max_abs_logit_diff":
            max_logit_diff.max().item(),

        "prediction_changes":
            int((~top1_agree).sum().item()),

        "fp32_time_s":
            time32,

        "fp16_time_s":
            time16,
    }

    results_dir = Path(
        "results"
    )
    results_dir.mkdir(
        exist_ok=True
    )

    summary_path = (
        results_dir
        / (
            f"{args.model}_imagenetv2_"
            f"quality_{args.tag}_summary.csv"
        )
    )

    with summary_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                summary.keys()
            ),
        )
        writer.writeheader()
        writer.writerow(
            summary
        )

    detail_path = (
        results_dir
        / (
            f"{args.model}_imagenetv2_"
            f"quality_{args.tag}_per_image.csv"
        )
    )

    with detail_path.open(
        "w",
        newline="",
    ) as f:
        fieldnames = [
            "path",
            "label",
            "pred_fp32",
            "pred_fp16",
            "correct_fp32",
            "correct_fp16",
            "top1_agree",
            "confidence_fp32",
            "confidence_fp16",
            "confidence_drift",
            "kl_fp32_to_fp16",
            "max_abs_logit_diff",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for i, (
            path,
            label,
        ) in enumerate(
            dataset.samples
        ):
            writer.writerow({
                "path": str(path),
                "label": label,
                "pred_fp32":
                    int(pred32[i]),
                "pred_fp16":
                    int(pred16[i]),
                "correct_fp32":
                    int(correct1_32[i]),
                "correct_fp16":
                    int(correct1_16[i]),
                "top1_agree":
                    int(top1_agree[i]),
                "confidence_fp32":
                    float(conf32[i]),
                "confidence_fp16":
                    float(conf16[i]),
                "confidence_drift":
                    float(conf_drift[i]),
                "kl_fp32_to_fp16":
                    float(per_image_kl[i]),
                "max_abs_logit_diff":
                    float(max_logit_diff[i]),
            })

    print("\nFORMAL QUALITY SUMMARY")
    print("-" * 72)

    for key, value in summary.items():
        print(
            f"{key:30s}: {value}"
        )

    print(
        "\nSaved:",
        summary_path,
    )

    print(
        "Saved:",
        detail_path,
    )


if __name__ == "__main__":
    main()
