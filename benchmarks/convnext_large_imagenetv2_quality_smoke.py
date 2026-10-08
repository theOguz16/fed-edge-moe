from pathlib import Path
import gc

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import convnext_large, ConvNeXt_Large_Weights


ROOT = Path(
    "data/imagenetv2/"
    "imagenetv2-matched-frequency-format-val"
)

NUM_CLASSES_SMOKE = 100
BATCH_SIZE = 4


class ImageNetV2NumericLabels(Dataset):
    def __init__(self, root, transform, num_classes=None):
        self.root = Path(root)
        self.transform = transform

        samples = []

        class_ids = sorted(
            [int(p.name) for p in self.root.iterdir() if p.is_dir()]
        )

        if num_classes is not None:
            class_ids = class_ids[:num_classes]

        # Smoke test: one deterministic image per class.
        for class_id in class_ids:
            class_dir = self.root / str(class_id)

            images = sorted(
                list(class_dir.glob("*.jpeg"))
                + list(class_dir.glob("*.jpg"))
                + list(class_dir.glob("*.png"))
            )

            if not images:
                raise RuntimeError(
                    f"No image found for class {class_id}"
                )

            samples.append((images[0], class_id))

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        with Image.open(path) as img:
            img = img.convert("RGB")
            img = self.transform(img)

        return img, label


def run_model(dtype, dataset, device):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    weights = ConvNeXt_Large_Weights.DEFAULT

    model = convnext_large(weights=weights)
    model.eval()
    model.to(device)

    if dtype == torch.float16:
        model.half()

    all_logits = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)

            if dtype == torch.float16:
                images = images.half()
            else:
                images = images.float()

            logits = model(images)

            # Quality comparison is done in FP32 on CPU.
            all_logits.append(
                logits.float().cpu()
            )
            all_labels.append(labels.cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    del model
    gc.collect()

    if device.type == "mps":
        torch.mps.empty_cache()

    return logits, labels


def accuracy(logits, labels, k=1):
    topk = logits.topk(k, dim=1).indices

    correct = (
        topk == labels.unsqueeze(1)
    ).any(dim=1)

    return correct.float().mean().item() * 100.0


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")

    device = torch.device("mps")

    weights = ConvNeXt_Large_Weights.DEFAULT
    transform = weights.transforms()

    dataset = ImageNetV2NumericLabels(
        ROOT,
        transform,
        num_classes=NUM_CLASSES_SMOKE,
    )

    print(
        f"Dataset: {len(dataset)} images, "
        f"labels {dataset.samples[0][1]}.."
        f"{dataset.samples[-1][1]}"
    )

    print("\nRunning FP32...")
    logits32, labels32 = run_model(
        torch.float32,
        dataset,
        device,
    )

    print("Running FP16...")
    logits16, labels16 = run_model(
        torch.float16,
        dataset,
        device,
    )

    assert torch.equal(labels32, labels16)

    labels = labels32

    top1_32 = accuracy(logits32, labels, 1)
    top5_32 = accuracy(logits32, labels, 5)

    top1_16 = accuracy(logits16, labels, 1)
    top5_16 = accuracy(logits16, labels, 5)

    pred32 = logits32.argmax(dim=1)
    pred16 = logits16.argmax(dim=1)

    top1_agreement = (
        pred32 == pred16
    ).float().mean().item() * 100.0

    top5_32_idx = logits32.topk(
        5, dim=1
    ).indices

    top5_16_idx = logits16.topk(
        5, dim=1
    ).indices

    overlaps = []

    for a, b in zip(top5_32_idx, top5_16_idx):
        set_a = set(a.tolist())
        set_b = set(b.tolist())

        overlaps.append(
            len(set_a & set_b) / 5.0
        )

    top5_overlap = (
        sum(overlaps) / len(overlaps) * 100.0
    )

    logp32 = F.log_softmax(
        logits32,
        dim=1,
    )

    p32 = logp32.exp()

    logp16 = F.log_softmax(
        logits16,
        dim=1,
    )

    kl_32_to_16 = F.kl_div(
        logp16,
        p32,
        reduction="batchmean",
    ).item()

    conf32 = F.softmax(
        logits32,
        dim=1,
    ).max(dim=1).values

    conf16 = F.softmax(
        logits16,
        dim=1,
    ).max(dim=1).values

    confidence_drift = (
        conf32 - conf16
    ).abs().mean().item()

    max_logit_diff = (
        logits32 - logits16
    ).abs().max().item()

    print("\nQUALITY SMOKE SUMMARY")
    print("-" * 60)

    print(
        f"FP32 top-1 accuracy : "
        f"{top1_32:7.3f}%"
    )

    print(
        f"FP16 top-1 accuracy : "
        f"{top1_16:7.3f}%"
    )

    print(
        f"FP32 top-5 accuracy : "
        f"{top5_32:7.3f}%"
    )

    print(
        f"FP16 top-5 accuracy : "
        f"{top5_16:7.3f}%"
    )

    print(
        f"Top-1 agreement     : "
        f"{top1_agreement:7.3f}%"
    )

    print(
        f"Top-5 set overlap   : "
        f"{top5_overlap:7.3f}%"
    )

    print(
        f"KL(FP32 || FP16)    : "
        f"{kl_32_to_16:.8f}"
    )

    print(
        f"Mean confidence drift: "
        f"{confidence_drift:.8f}"
    )

    print(
        f"Max abs logit diff  : "
        f"{max_logit_diff:.8f}"
    )


if __name__ == "__main__":
    main()
