import torch

from ml.model.moe import SparseMoE


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_device()

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
    requires_grad=True,
)

moe = SparseMoE(
    d_model=d_model,
    num_experts=num_experts,
    top_k=top_k,
    expert_hidden_dim=256,
).to(device)

output = moe(x)

loss = output.hidden_states.pow(2).mean() + 0.01 * output.aux_loss

loss.backward()

print("Device:", device)

print()
print("Input shape :", x.shape)
print("Output shape:", output.hidden_states.shape)

print()
print("Expert counts:")
print(output.expert_counts.detach().cpu())

print()
print(
    "Total expert routes:",
    int(output.expert_counts.sum().detach().cpu())
)

print(
    "Expected routes:",
    batch_size * seq_len * top_k
)

print()
print(
    "Aux loss:",
    float(output.aux_loss.detach().cpu())
)

print(
    "Training loss:",
    float(loss.detach().cpu())
)

assert output.hidden_states.shape == x.shape

assert int(output.expert_counts.sum().detach().cpu()) == (
    batch_size * seq_len * top_k
)

assert x.grad is not None

print()
print("SparseMoE forward OK")
print("SparseMoE backward OK")
print("Top-K dispatch OK")
print("Weighted expert combination OK")
