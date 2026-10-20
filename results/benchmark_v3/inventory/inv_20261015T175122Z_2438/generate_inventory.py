import csv
import gc
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
import torchvision
from torch import nn
from torchvision.models import resnet50, convnext_base, convnext_large

torch.set_num_threads(4)
outdir = Path(sys.argv[1])

specs = [
    ("resnet50", resnet50, "resnet50__stage6.json",
     [(1,1),(2,2),(3,3),(4,4),(5,5),(6,6)]),
    ("convnext_base", convnext_base, "convnext_base__fine41.json",
     [(1,1),(2,4),(5,8),(9,36),(37,40),(41,41)]),
    ("convnext_large", convnext_large, "convnext_large__fine41.json",
     [(1,1),(2,4),(5,8),(9,36),(37,40),(41,41)]),
]

def stages(m, name):
    if name == "resnet50":
        return [
            nn.Sequential(m.conv1,m.bn1,m.relu,m.maxpool),
            m.layer1,m.layer2,m.layer3,m.layer4,
            nn.Sequential(m.avgpool,nn.Flatten(1),m.fc),
        ]
    return [
        m.features[0],m.features[1],
        nn.Sequential(m.features[2],m.features[3]),
        nn.Sequential(m.features[4],m.features[5]),
        nn.Sequential(m.features[6],m.features[7]),
        nn.Sequential(m.avgpool,m.classifier),
    ]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

rows = []
manifest_hashes = {}

for name, factory, manifest_name, ranges in specs:
    manifest_path = (
        Path("configs/partition_manifests_v2") / manifest_name
    )
    manifest = json.loads(manifest_path.read_text())
    manifest_hashes[name] = sha(manifest_path)
    model = factory(weights=None).eval().cpu()
    groups = stages(model, name)
    x = torch.zeros(1,3,224,224)
    seen = set()
    seen_buffers = set()
    model_macs = 0
    model_params = 0

    with torch.inference_mode():
        for i, (stage, bounds) in enumerate(zip(groups,ranges)):
            a,b = bounds
            paths = [
                path
                for group in manifest["groups"][a-1:b]
                for path in group["modules"]
            ]

            params = [
                p for p in stage.parameters()
                if id(p) not in seen
            ]
            seen.update(id(p) for p in params)

            buffers = [
                buf for buf in stage.buffers()
                if id(buf) not in seen_buffers
            ]
            seen_buffers.update(id(buf) for buf in buffers)

            param_count = sum(p.numel() for p in params)
            param_bytes = sum(
                p.numel()*p.element_size() for p in params
            )
            buffer_bytes = sum(
                t.numel()*t.element_size() for t in buffers
            )

            counter = [0]

            def hook(module, inputs, output):
                if isinstance(module, nn.Conv2d):
                    kh,kw = module.kernel_size
                    counter[0] += (
                        output.numel()
                        * (module.in_channels//module.groups)
                        * kh*kw
                    )
                elif isinstance(module, nn.Linear):
                    counter[0] += (
                        output.numel()*module.in_features
                    )

            hooks = [
                m.register_forward_hook(hook)
                for m in stage.modules()
                if isinstance(m,(nn.Conv2d,nn.Linear))
            ]

            input_shape = list(x.shape)
            try:
                x = stage(x)
            finally:
                for h in hooks:
                    h.remove()

            macs = int(counter[0])
            model_macs += macs
            model_params += param_count

            rows.append({
                "model":name,
                "stage_id":f"S{i}",
                "manifest":manifest_name,
                "manifest_sha256":manifest_hashes[name],
                "source_groups":f"G{a}-G{b}",
                "module_paths":json.dumps(paths),
                "input_shape":json.dumps(input_shape),
                "output_shape":json.dumps(list(x.shape)),
                "parameter_count":param_count,
                "parameter_bytes":param_bytes,
                "buffer_bytes":buffer_bytes,
                "conv_linear_macs":macs,
                "conv_linear_flops":2*macs,
                "output_activation_bytes":
                    x.numel()*x.element_size(),
            })

    assert model_params == sum(
        p.numel() for p in model.parameters()
    ), f"Parameter coverage failed: {name}"

    print(
        f"{name}: 6 stages, "
        f"{model_params:,} params, "
        f"{model_macs/1e9:.4f} G MACs — PASS",
        flush=True
    )

    del x,groups,model
    gc.collect()

assert len(rows)==18

csv_path = outdir/"structural_inventory.csv"
with csv_path.open("w",newline="") as f:
    writer = csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

metadata = {
    "protocol":"structural_inventory_v3",
    "input_shape":[1,3,224,224],
    "dtype":"float32",
    "weight_loading":"weights=None; structural analysis only",
    "mac_scope":"Conv2d and Linear",
    "flop_convention":"2 FLOPs per MAC",
    "torch":torch.__version__,
    "torchvision":torchvision.__version__,
    "git_commit":subprocess.check_output(
        ["git","rev-parse","HEAD"],text=True
    ).strip(),
    "git_dirty":bool(subprocess.check_output(
        ["git","status","--porcelain"],text=True
    ).strip()),
    "manifest_sha256":manifest_hashes,
    "stage_rows":len(rows),
}

meta_path=outdir/"metadata.json"
meta_path.write_text(json.dumps(metadata,indent=2))

checksums={
    p.name:sha(p) for p in (
        Path(__file__),csv_path,meta_path
    )
}
(outdir/"checksums.json").write_text(
    json.dumps(checksums,indent=2)
)

print("\nSTAGE ROWS:",len(rows))
print("STRUCTURAL INVENTORY: PASS")
print("RESULT DIRECTORY:",outdir)
