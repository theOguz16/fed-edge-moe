import csv
import math
import statistics
from pathlib import Path


INPUT = Path(
    "results/"
    "convnext_large_vision_hard_msi_"
    "pipeline_power.csv"
)

SUMMARY_OUT = Path(
    "results/"
    "convnext_large_vision_hard_msi_"
    "pipeline_operating_points.csv"
)

FEASIBILITY_OUT = Path(
    "results/"
    "convnext_large_vision_hard_msi_"
    "pipeline_memory_feasibility.csv"
)


THROUGHPUT_TARGETS = [
    1.5,
    2.0,
    5.0,
    8.0,
    10.0,
    14.0,
]


with INPUT.open(
    newline="",
) as f:
    raw = list(
        csv.DictReader(f)
    )


configs = sorted(
    set(
        r["config"]
        for r in raw
    )
)


def median_for(
    rows,
    key,
):
    return statistics.median(
        float(r[key])
        for r in rows
    )


summary = []


for config in configs:

    group = [
        r for r in raw
        if r["config"] == config
    ]

    row = {
        "config":
            config,

        "throughput_img_s":
            median_for(
                group,
                "throughput_img_s",
            ),

        "median_latency_ms":
            median_for(
                group,
                "median_latency_ms",
            ),

        "compute_power_w":
            median_for(
                group,
                "mean_compute_power_w",
            ),

        "compute_energy_img_j":
            median_for(
                group,
                "compute_energy_img_j",
            ),

        "cuda_allocated_mb":
            median_for(
                group,
                "torch_cuda_allocated_mb",
            ),

        "cuda_peak_mb":
            median_for(
                group,
                "torch_cuda_peak_mb",
            ),

        "nvml_peak_used_mb":
            median_for(
                group,
                "nvml_peak_used_mb",
            ),
    }

    summary.append(row)


def is_dominated(
    candidate,
    pool,
):
    """
    Higher throughput is better.
    Lower energy/image is better.

    Memory is handled separately
    as a hard feasibility constraint.
    """

    for other in pool:

        if (
            other["config"]
            == candidate["config"]
        ):
            continue

        no_worse = (
            other["throughput_img_s"]
            >=
            candidate["throughput_img_s"]
            and
            other["compute_energy_img_j"]
            <=
            candidate["compute_energy_img_j"]
        )

        strictly_better = (
            other["throughput_img_s"]
            >
            candidate["throughput_img_s"]
            or
            other["compute_energy_img_j"]
            <
            candidate["compute_energy_img_j"]
        )

        if (
            no_worse
            and
            strictly_better
        ):
            return True

    return False


# Global Pareto status without
# a memory constraint.
for row in summary:
    row["global_pareto"] = (
        not is_dominated(
            row,
            summary,
        )
    )


# Add fixed budgets plus actual
# experiment-derived breakpoints.
budgets = {
    0.0,
    250.0,
    500.0,
    750.0,
    800.0,
}


for row in summary:

    peak = row[
        "cuda_peak_mb"
    ]

    rounded = (
        math.ceil(
            peak / 10.0
        )
        * 10.0
    )

    budgets.add(
        rounded
    )


budgets = sorted(
    budgets
)


SUMMARY_OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with SUMMARY_OUT.open(
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=
            summary[0].keys(),
    )

    writer.writeheader()
    writer.writerows(
        summary
    )


print(
    "\nGLOBAL OPERATING POINTS"
)

print(
    "-" * 125
)


for row in sorted(
    summary,
    key=lambda x:
        x["cuda_peak_mb"],
):

    status = (
        "PARETO"
        if row["global_pareto"]
        else "dominated"
    )

    print(
        f"{row['config']:12} | "
        f"thr "
        f"{row['throughput_img_s']:6.2f} | "
        f"lat "
        f"{row['median_latency_ms']:7.1f} ms | "
        f"P "
        f"{row['compute_power_w']:5.1f} W | "
        f"E "
        f"{row['compute_energy_img_j']:6.3f} J/img | "
        f"alloc "
        f"{row['cuda_allocated_mb']:7.1f} MB | "
        f"peak "
        f"{row['cuda_peak_mb']:7.1f} MB | "
        f"{status}"
    )


feasibility_rows = []


print(
    "\n\nMEMORY-AWARE FEASIBILITY"
)


for budget in budgets:

    feasible = [
        row
        for row in summary
        if (
            row["cuda_peak_mb"]
            <= budget
            + 1e-9
        )
    ]

    print(
        "\n"
        + "=" * 90
    )

    print(
        f"CUDA peak budget "
        f"<= {budget:.0f} MB"
    )

    print(
        "=" * 90
    )


    if not feasible:

        print(
            "No feasible configuration."
        )

        continue


    pareto_configs = []


    for row in feasible:

        dominated = is_dominated(
            row,
            feasible,
        )

        if not dominated:
            pareto_configs.append(
                row["config"]
            )

        feasibility_rows.append({
            "memory_budget_mb":
                budget,

            "config":
                row["config"],

            "feasible":
                1,

            "pareto_within_budget":
                int(
                    not dominated
                ),

            "throughput_img_s":
                row[
                    "throughput_img_s"
                ],

            "median_latency_ms":
                row[
                    "median_latency_ms"
                ],

            "compute_power_w":
                row[
                    "compute_power_w"
                ],

            "compute_energy_img_j":
                row[
                    "compute_energy_img_j"
                ],

            "cuda_peak_mb":
                row[
                    "cuda_peak_mb"
                ],
        })


    for row in sorted(
        feasible,
        key=lambda x:
            (
                -x[
                    "throughput_img_s"
                ],
                x[
                    "compute_energy_img_j"
                ],
            ),
    ):

        status = (
            "PARETO"
            if row["config"]
            in pareto_configs
            else "dominated"
        )

        print(
            f"{row['config']:12} | "
            f"{row['throughput_img_s']:6.2f} img/s | "
            f"{row['compute_energy_img_j']:6.3f} J/img | "
            f"{row['cuda_peak_mb']:7.1f} MB | "
            f"{status}"
        )


    print(
        "\nMinimum-energy choice "
        "under throughput constraints:"
    )


    for target in (
        THROUGHPUT_TARGETS
    ):

        candidates = [
            row
            for row in feasible
            if (
                row[
                    "throughput_img_s"
                ]
                >= target
            )
        ]


        if not candidates:

            print(
                f"  >= {target:4.1f} img/s "
                f"-> infeasible"
            )

            continue


        best = min(
            candidates,
            key=lambda x:
                (
                    x[
                        "compute_energy_img_j"
                    ],
                    -x[
                        "throughput_img_s"
                    ],
                ),
        )


        print(
            f"  >= {target:4.1f} img/s "
            f"-> {best['config']:12} | "
            f"{best['throughput_img_s']:6.2f} img/s | "
            f"{best['compute_energy_img_j']:6.3f} J/img"
        )


if feasibility_rows:

    with FEASIBILITY_OUT.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=
                feasibility_rows[0].keys(),
        )

        writer.writeheader()

        writer.writerows(
            feasibility_rows
        )


print(
    "\nSaved:"
)

print(
    SUMMARY_OUT
)

print(
    FEASIBILITY_OUT
)
