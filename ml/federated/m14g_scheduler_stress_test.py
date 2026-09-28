SCENARIOS = [
    {
        "name": "normal",
        "required_memory_gb": 3.5,
        "mac_available": True,
        "msi_available": True,
    },
    {
        "name": "mac_busy",
        "required_memory_gb": 3.5,
        "mac_available": False,
        "msi_available": True,
    },
    {
        "name": "large_expert",
        "required_memory_gb": 6.0,
        "mac_available": True,
        "msi_available": True,
    },
    {
        "name": "no_device",
        "required_memory_gb": 6.0,
        "mac_available": False,
        "msi_available": True,
    },
]

BASE = {
    "mac_m4": {
        "memory": 16.0,
        "affinity": 0.214,
        "seconds": 3.462,
    },
    "msi_rtx3050": {
        "memory": 4.0,
        "affinity": 0.191,
        "seconds": 3.458,
    },
}


def choose(scenario):
    candidates = []

    for name, d in BASE.items():
        available = (
            scenario["mac_available"]
            if name == "mac_m4"
            else scenario["msi_available"]
        )

        if not available:
            continue

        if (
            d["memory"]
            < scenario["required_memory_gb"]
        ):
            continue

        candidates.append({
            "name": name,
            "score":
                d["affinity"]
                / d["seconds"],
        })

    if not candidates:
        return "DEFER_TASK"

    return max(
        candidates,
        key=lambda x: x["score"],
    )["name"]


def main():
    print()
    print("FedEdgeMoE - M14G")
    print("Scheduler Stress Test")
    print()

    for scenario in SCENARIOS:
        winner = choose(scenario)

        print(
            f"{scenario['name']:<14} "
            f"| required="
            f"{scenario['required_memory_gb']:.1f}GB "
            f"| decision={winner}"
        )


if __name__ == "__main__":
    main()
