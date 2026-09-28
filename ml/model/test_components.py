import torch

from ml.model.expert import ExpertMLP
from ml.model.router import TopKRouter


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_device()

print("Device:", device)

batch_size = 2
seq_len = 4
d_model = 128

num_experts = 8
top_k = 2

x = torch.randn(
    batch_size,
    seq_len,
    d_model,
    device=device,
)

expert = ExpertMLP(
    d_model=d_model,
    hidden_dim=256,
).to(device)

expert_output = expert(x)

print()
print("Expert input :", x.shape)
print("Expert output:", expert_output.shape)

router = TopKRouter(
    d_model=d_model,
    num_experts=num_experts,
    top_k=top_k,
).to(device)

route = router(x)

print()
print("Top-K indices shape:", route.topk_indices.shape)
print("Top-K weights shape:", route.topk_weights.shape)

print()
print("Ilk token expert secimi:")
print(route.topk_indices[0, 0].detach().cpu())

print("Ilk token expert agirliklari:")
print(route.topk_weights[0, 0].detach().cpu())

print()
print("Expert kullanim sayilari:")
print(route.expert_counts.detach().cpu())

print()
print("Aux loss:", float(route.aux_loss.detach().cpu()))

assert expert_output.shape == x.shape
assert route.topk_indices.shape == (batch_size, seq_len, top_k)

print()
print("✓ ExpertMLP OK")
print("✓ TopKRouter OK")
