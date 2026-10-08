import torch
from torchvision.models import (
    convnext_large,
    ConvNeXt_Large_Weights,
)


RESOLUTION = 320
BATCH = 1


def mb(n):
    return n / (1024 ** 2)


model = convnext_large(
    weights=ConvNeXt_Large_Weights.DEFAULT
)
model.eval()

x = torch.randn(
    BATCH,
    3,
    RESOLUTION,
    RESOLUTION,
)

print("MODEL FEATURES")
print("=" * 90)

current = x

with torch.inference_mode():

    for i, layer in enumerate(model.features):

        current = layer(current)

        elements = current.numel()
        bytes_fp32 = elements * 4
        bytes_fp16 = elements * 2

        params = sum(
            p.numel()
            for p in layer.parameters()
        )

        print(
            f"features[{i}] "
            f"{layer.__class__.__name__:20} | "
            f"shape={tuple(current.shape)!s:24} | "
            f"params={params / 1e6:8.2f} M | "
            f"act FP32={mb(bytes_fp32):7.2f} MB | "
            f"FP16={mb(bytes_fp16):7.2f} MB"
        )

    pooled = model.avgpool(current)

    print("\nAFTER AVGPOOL")
    print(tuple(pooled.shape))

    output = model.classifier(pooled)

    print("OUTPUT")
    print(tuple(output.shape))


print("\nCANDIDATE CUT POINTS")
print("=" * 90)

cut_names = {
    1: "after_stage1",
    3: "after_stage2",
    5: "after_stage3",
    7: "after_stage4",
}

current = x

with torch.inference_mode():

    for i, layer in enumerate(model.features):
        current = layer(current)

        if i in cut_names:

            fp32_mb = (
                current.numel()
                * current.element_size()
                / 1024**2
            )

            print(
                f"{cut_names[i]:18} | "
                f"after features[{i}] | "
                f"shape={tuple(current.shape)} | "
                f"transfer FP32={fp32_mb:.2f} MB"
            )


print("\nSPLIT CORRECTNESS")
print("=" * 90)

with torch.inference_mode():

    reference = model(x)

    for cut in [1, 3, 5]:

        left = model.features[
            : cut + 1
        ]

        right = model.features[
            cut + 1 :
        ]

        z = left(x)
        z = right(z)
        z = model.avgpool(z)
        split_output = model.classifier(z)

        max_abs = (
            split_output
            - reference
        ).abs().max().item()

        mean_abs = (
            split_output
            - reference
        ).abs().mean().item()

        print(
            f"cut after features[{cut}] | "
            f"max_abs_diff={max_abs:.8e} | "
            f"mean_abs_diff={mean_abs:.8e}"
        )
