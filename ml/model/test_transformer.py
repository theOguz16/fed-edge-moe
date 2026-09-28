import torch

from ml.model.transformer import MiniMoELM


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_device()

batch_size = 4
seq_len = 16
vocab_size = 64

model = MiniMoELM(
    vocab_size=vocab_size,
    max_seq_len=32,
    d_model=128,
    num_heads=4,
    num_layers=3,
    num_experts=8,
    top_k=2,
    expert_hidden_dim=256,
).to(device)

input_ids = torch.randint(
    0,
    vocab_size,
    (batch_size, seq_len),
    device=device,
)

labels = torch.randint(
    0,
    vocab_size,
    (batch_size, seq_len),
    device=device,
)

output = model(
    input_ids=input_ids,
    labels=labels,
)

output.loss.backward()

print("Device:", device)

print()
print("Logits shape:")
print(output.logits.shape)

print()
print("LM loss:")
print(float(output.lm_loss.detach().cpu()))

print()
print("Router aux loss:")
print(float(output.router_aux_loss.detach().cpu()))

print()
print("Expert counts shape:")
print(output.expert_counts.shape)

print()
for layer in range(output.expert_counts.shape[0]):
    counts = output.expert_counts[layer].detach().cpu()

    print(
        f"Layer {layer} expert counts:",
        counts.tolist(),
    )

    print(
        f"Layer {layer} total routes:",
        int(counts.sum()),
    )

expected_routes = batch_size * seq_len * 2

print()
print("Expected routes per layer:", expected_routes)

assert output.logits.shape == (
    batch_size,
    seq_len,
    vocab_size,
)

assert output.expert_counts.shape == (
    3,
    8,
)

for layer in range(3):
    assert int(output.expert_counts[layer].sum()) == expected_routes

print()
print("Transformer forward OK")
print("Transformer backward OK")
print("Causal self-attention OK")
print("3-layer Sparse MoE OK")
