FAST_AFFINITY = 0.20
FAST_SECONDS = 4.06

PARTNER_AFFINITY_LIMIT = 0.40

CANDIDATES = {
    "edge_5s": 5.0,
    "edge_6s": 6.0,
    "edge_8s": 8.0,
    "edge_10s": 10.0,
    "edge_11_24s": 11.24,
    "edge_14s": 14.0,
}


fast_efficiency = (
    FAST_AFFINITY
    / FAST_SECONDS
)

max_viable_seconds = (
    PARTNER_AFFINITY_LIMIT
    / fast_efficiency
)

print()
print("FedEdgeMoE - M12K")
print("Cost-Aware Candidate Pre-Filter")
print()

print(
    "Maximum viable round time:",
    f"{max_viable_seconds:.2f}s",
)

print()

for name, seconds in CANDIDATES.items():
    required_affinity = (
        fast_efficiency
        * seconds
    )

    viable = (
        required_affinity
        <= PARTNER_AFFINITY_LIMIT
    )

    print(
        f"{name:<14} "
        f"| {seconds:5.2f}s "
        f"| required="
        f"{required_affinity*100:5.1f}% "
        f"| {'KEEP' if viable else 'PRUNE'}"
    )
