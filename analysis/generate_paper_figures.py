from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "paper" / "figures"
DATA_OUT = OUT / "data"

OUT.mkdir(parents=True, exist_ok=True)
DATA_OUT.mkdir(parents=True, exist_ok=True)

UNIFIED = RESULTS / "unified_characterization.csv"
CROSS_SHORT = RESULTS / "crossover_analysis_short.csv"

if not UNIFIED.exists():
    raise FileNotFoundError(f"Missing: {UNIFIED}")


# ============================================================
# Style
# ============================================================

sns.set_theme(style="whitegrid", context="paper")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "Apple M4": "#0072B2",
    "Intel i7-11800H": "#009E73",
    "RTX 3050 Laptop": "#D55E00",
    "fp32": "#4C78A8",
    "fp16": "#F58518",
}

BOUNDARY_LABELS = {
    "combined_soc": "Combined SoC",
    "cpu_package": "CPU package",
    "gpu_board": "GPU board",
}


def save(fig, stem):
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[OK] {stem}.pdf / .png")


def require_rows(frame, name):
    if frame.empty:
        raise RuntimeError(f"No rows found for figure: {name}")
    print(f"[DATA] {name}: {len(frame)} rows")


def num(s):
    return pd.to_numeric(s, errors="coerce")


df = pd.read_csv(UNIFIED)

for col in [
    "throughput",
    "threads",
    "batch",
    "resolution",
    "power_w",
    "energy_per_item_j",
    "memory_allocated_mb",
    "memory_peak_mb",
]:
    if col in df.columns:
        df[col] = num(df[col])


# ============================================================
# Figure 1 — Device crossover
# Median MSI/Mac throughput ratio from matched short runs.
# >1 means MSI-side accelerator wins, <1 means Apple M4 wins.
# ============================================================

def build_crossover():
    if not CROSS_SHORT.exists():
        raise FileNotFoundError(
            "results/crossover_analysis_short.csv is required for Fig. 1"
        )

    c = pd.read_csv(CROSS_SHORT)

    # Accept several possible column names so the plotting script
    # survives small analysis-script naming changes.
    ratio_candidates = [
        "msi_over_mac",
        "speedup_msi_over_mac",
        "throughput_ratio",
        "ratio",
    ]

    ratio_col = next((x for x in ratio_candidates if x in c.columns), None)

    if ratio_col is None:
        mac_candidates = [
            "mac_throughput",
            "apple_throughput",
            "throughput_mac",
        ]
        msi_candidates = [
            "msi_throughput",
            "rtx_throughput",
            "throughput_msi",
        ]

        mac_col = next((x for x in mac_candidates if x in c.columns), None)
        msi_col = next((x for x in msi_candidates if x in c.columns), None)

        if mac_col is None or msi_col is None:
            raise RuntimeError(
                "Could not identify throughput ratio columns in "
                f"{CROSS_SHORT.name}.\nColumns: {list(c.columns)}"
            )

        c["_ratio"] = num(c[msi_col]) / num(c[mac_col])
        ratio_col = "_ratio"

    if "model" not in c.columns:
        raise RuntimeError(
            f"'model' column missing from {CROSS_SHORT.name}"
        )

    c[ratio_col] = num(c[ratio_col])
    c = c.dropna(subset=["model", ratio_col])

    summary = (
        c.groupby("model", as_index=False)[ratio_col]
        .median()
        .rename(columns={ratio_col: "median_msi_over_mac"})
    )

    # Research-story order instead of alphabetical soup.
    order = [
        "DistilGPT2",
        "Qwen2.5-1.5B-Instruct",
        "Qwen3-1.7B",
        "ResNet50",
        "ConvNeXt-Base",
        "ConvNeXt-Large",
    ]
    summary["order"] = summary["model"].map(
        {m: i for i, m in enumerate(order)}
    )
    summary = summary.sort_values(["order", "model"])

    summary["display_model"] = summary["model"].replace({
        "Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B",
        "Qwen3-1.7B": "Qwen3-1.7B",
    })

    require_rows(summary, "Fig. 1 crossover")
    summary.to_csv(DATA_OUT / "fig1_device_crossover.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.8, 3.15))

    colors = [
        "#D55E00" if x >= 1 else "#0072B2"
        for x in summary["median_msi_over_mac"]
    ]

    bars = ax.bar(
        summary["display_model"],
        summary["median_msi_over_mac"],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("MSI / Apple M4 throughput ratio\n(median, matched short runs)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)

    for bar, value in zip(bars, summary["median_msi_over_mac"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(0.025, value * 0.015),
            f"{value:.2f}×",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.text(
        0.99,
        0.03,
        "1.0 = equal throughput",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="0.35",
    )

    save(fig, "fig1_device_crossover")


# ============================================================
# Figure 2 — Thread scaling
# Same hard vision workload on both CPUs:
# ConvNeXt-Large, 320x320, batch=1.
# ============================================================

def build_thread_scaling():
    d = df[
        (df["model"] == "ConvNeXt-Large")
        & (df["backend"] == "cpu")
        & (df["measurement_mode"] == "cpu_scaling_short")
        & (df["resolution"] == 320)
        & (df["batch"] == 1)
        & (df["device"].isin(["Apple M4", "Intel i7-11800H"]))
    ].copy()

    d = d.dropna(subset=["threads", "throughput"])
    d = (
        d.groupby(["device", "threads"], as_index=False)["throughput"]
        .median()
    )

    require_rows(d, "Fig. 2 thread scaling")
    d.to_csv(DATA_OUT / "fig2_thread_scaling.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.8, 3.35))

    for device in ["Apple M4", "Intel i7-11800H"]:
        part = d[d["device"] == device].sort_values("threads")
        if part.empty:
            continue

        ax.plot(
            part["threads"],
            part["throughput"],
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            label=device,
            color=COLORS[device],
        )

        # Mark empirical optimum.
        best = part.loc[part["throughput"].idxmax()]
        ax.scatter(
            [best["threads"]],
            [best["throughput"]],
            s=58,
            facecolors="none",
            edgecolors=COLORS[device],
            linewidths=1.5,
            zorder=5,
        )

        ax.annotate(
            f'best: {int(best["threads"])} threads',
            xy=(best["threads"], best["throughput"]),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=7,
            color=COLORS[device],
        )

    ax.set_xlabel("CPU threads")
    ax.set_ylabel("Throughput (images/s)")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.15)

    save(fig, "fig2_thread_scaling")


# ============================================================
# Helper — sustained accelerator rows for ConvNeXt-Large
# Used by Figures 3, 4 and 5.
# ============================================================

def convnext_large_sustained_accel():
    d = df[
        (df["model"] == "ConvNeXt-Large")
        & (df["measurement_mode"] == "sustained_power")
        & (df["device"].isin(["Apple M4", "RTX 3050 Laptop"]))
        & (df["backend"].isin(["mps", "cuda"]))
        & (df["precision"].isin(["fp32", "fp16"]))
        & (df["profile"].isin(["light", "medium", "heavy"]))
    ].copy()

    return d


# ============================================================
# Figure 3 — Precision vs throughput
# Sustained ConvNeXt-Large representative profiles.
# Separate panels avoid implying identical device behavior.
# ============================================================

def build_precision_throughput():
    d = convnext_large_sustained_accel()
    d = d.dropna(subset=["throughput"])

    d = (
        d.groupby(
            ["device", "profile", "precision"],
            as_index=False
        )["throughput"]
        .median()
    )

    require_rows(d, "Fig. 3 precision throughput")
    d.to_csv(
        DATA_OUT / "fig3_precision_throughput.csv",
        index=False
    )

    profiles = ["light", "medium", "heavy"]
    precisions = ["fp32", "fp16"]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(6.8, 3.05),
        sharey=False
    )

    for ax, device in zip(
        axes,
        ["Apple M4", "RTX 3050 Laptop"]
    ):
        p = d[d["device"] == device]

        x = np.arange(len(profiles))
        width = 0.36

        for i, precision in enumerate(precisions):
            vals = []
            for profile in profiles:
                row = p[
                    (p["profile"] == profile)
                    & (p["precision"] == precision)
                ]
                vals.append(
                    row["throughput"].iloc[0]
                    if not row.empty else np.nan
                )

            ax.bar(
                x + (i - 0.5) * width,
                vals,
                width=width,
                label=precision.upper(),
                color=COLORS[precision],
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [x.capitalize() for x in profiles]
        )
        ax.set_title(device)
        ax.set_xlabel("Workload profile")
        ax.set_ylabel("Throughput (images/s)")

    axes[1].legend(frameon=False)
    save(fig, "fig3_precision_throughput")


# ============================================================
# Figure 4 — Precision vs accelerator memory
# Uses memory_allocated_mb so the plotted quantity is kept
# consistent between the Apple and NVIDIA accelerator rows.
# ============================================================

def build_memory():
    # Use matched accelerator configurations where BOTH devices
    # report framework-allocated accelerator memory.
    #
    # Representative workload mapping:
    # light  = 160x160, batch 1
    # medium = 224x224, batch 4
    # heavy  = 320x320, batch 16

    d = df[
        (df["model"] == "ConvNeXt-Large")
        & (df["measurement_mode"] == "short_sweep")
        & (df["device"].isin(["Apple M4", "RTX 3050 Laptop"]))
        & (df["backend"].isin(["mps", "cuda"]))
        & (df["precision"].isin(["fp32", "fp16"]))
        & (df["memory_allocated_mb"].notna())
    ].copy()

    light = (d["resolution"] == 160) & (d["batch"] == 1)
    medium = (d["resolution"] == 224) & (d["batch"] == 4)
    heavy = (d["resolution"] == 320) & (d["batch"] == 16)

    d = d[light | medium | heavy].copy()

    d["profile_plot"] = np.select(
        [
            (d["resolution"] == 160) & (d["batch"] == 1),
            (d["resolution"] == 224) & (d["batch"] == 4),
            (d["resolution"] == 320) & (d["batch"] == 16),
        ],
        ["light", "medium", "heavy"],
        default="unknown",
    )

    d = (
        d.groupby(
            ["device", "profile_plot", "precision"],
            as_index=False
        )["memory_allocated_mb"]
        .median()
    )

    require_rows(d, "Fig. 4 memory")
    d.to_csv(
        DATA_OUT / "fig4_memory_usage.csv",
        index=False
    )

    profiles = ["light", "medium", "heavy"]
    precisions = ["fp32", "fp16"]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(6.8, 3.05),
        sharey=True
    )

    for ax, device in zip(
        axes,
        ["Apple M4", "RTX 3050 Laptop"]
    ):
        p = d[d["device"] == device]

        x = np.arange(len(profiles))
        width = 0.36

        for i, precision in enumerate(precisions):
            vals = []

            for profile in profiles:
                row = p[
                    (p["profile_plot"] == profile)
                    & (p["precision"] == precision)
                ]

                vals.append(
                    row["memory_allocated_mb"].iloc[0]
                    if not row.empty else np.nan
                )

            bars = ax.bar(
                x + (i - 0.5) * width,
                vals,
                width=width,
                label=precision.upper(),
                color=COLORS[precision],
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            ["Light\n160/B1", "Medium\n224/B4", "Heavy\n320/B16"]
        )
        ax.set_title(device)
        ax.set_xlabel("Representative configuration")
        ax.set_ylabel("Allocated accelerator memory (MB)")

    axes[1].legend(frameon=False)

    fig.text(
        0.5,
        -0.04,
        "Framework-reported allocated accelerator memory; "
        "this is not whole-system RAM usage.",
        ha="center",
        fontsize=7,
        color="0.35",
    )

    fig.subplots_adjust(bottom=0.23)

    save(fig, "fig4_memory_usage")


# ============================================================
# Figure 5 — Sustained power
# IMPORTANT: Mac and RTX are shown in separate panels because
# energy boundaries differ:
# Mac = combined SoC
# RTX = GPU board
# Absolute bars must not be interpreted as same-boundary power.
# ============================================================

def build_power():
    d = convnext_large_sustained_accel()
    d = d.dropna(subset=["power_w"])

    d = (
        d.groupby(
            ["device", "profile", "precision", "energy_boundary"],
            dropna=False,
            as_index=False
        )["power_w"]
        .median()
    )

    require_rows(d, "Fig. 5 power")
    d.to_csv(DATA_OUT / "fig5_power_consumption.csv", index=False)

    profiles = ["light", "medium", "heavy"]
    precisions = ["fp32", "fp16"]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(6.8, 3.15),
        sharey=False
    )

    for ax, device in zip(
        axes,
        ["Apple M4", "RTX 3050 Laptop"]
    ):
        p = d[d["device"] == device]

        x = np.arange(len(profiles))
        width = 0.36

        for i, precision in enumerate(precisions):
            vals = []
            for profile in profiles:
                row = p[
                    (p["profile"] == profile)
                    & (p["precision"] == precision)
                ]
                vals.append(
                    row["power_w"].iloc[0]
                    if not row.empty else np.nan
                )

            ax.bar(
                x + (i - 0.5) * width,
                vals,
                width=width,
                label=precision.upper(),
                color=COLORS[precision],
            )

        boundary_values = [
            str(x)
            for x in p["energy_boundary"].dropna().unique()
        ]
        boundary = (
            boundary_values[0]
            if boundary_values
            else "measurement boundary unavailable"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [x.capitalize() for x in profiles]
        )
        ax.set_title(f"{device}\n({BOUNDARY_LABELS.get(boundary, boundary)})")
        ax.set_xlabel("Workload profile")
        ax.set_ylabel("Average power (W)")

    axes[1].legend(frameon=False)

    fig.text(
        0.5,
        -0.01,
        "Power boundaries differ across devices; "
        "absolute cross-device values are not directly comparable.",
        ha="center",
        fontsize=7,
        color="0.35",
    )

    fig.subplots_adjust(bottom=0.20)
    save(fig, "fig5_power_consumption")


# ============================================================
# Figure 6 — Speed-energy trade-off
# Vision only because tok/s and img/s must not be mixed.
#
# Separate panels by device/boundary. Each point represents a
# sustained configuration. The desirable direction is:
# higher throughput (right), lower J/image (down).
# ============================================================

def build_energy_tradeoff():
    d = df[
        (df["modality"] == "vision")
        & (df["measurement_mode"] == "sustained_power")
        & (df["throughput"].notna())
        & (df["energy_per_item_j"].notna())
        & (
            df["device"].isin([
                "Apple M4",
                "Intel i7-11800H",
                "RTX 3050 Laptop",
            ])
        )
    ].copy()

    d = d[
        (d["throughput"] > 0)
        & (d["energy_per_item_j"] > 0)
    ]

    require_rows(d, "Fig. 6 energy trade-off")

    keep = [
        "model",
        "device",
        "backend",
        "precision",
        "profile",
        "resolution",
        "batch",
        "threads",
        "throughput",
        "energy_per_item_j",
        "power_w",
        "energy_boundary",
    ]
    d[keep].to_csv(
        DATA_OUT / "fig6_speed_energy_tradeoff.csv",
        index=False
    )

    devices = [
        "Apple M4",
        "Intel i7-11800H",
        "RTX 3050 Laptop",
    ]

    model_markers = {
        "ResNet50": "o",
        "ConvNeXt-Base": "s",
        "ConvNeXt-Large": "^",
    }

    fig, axes = plt.subplots(
        1, 3,
        figsize=(7.15, 2.95),
        sharex=False,
        sharey=False
    )

    for ax, device in zip(axes, devices):
        p = d[d["device"] == device]

        for model, marker in model_markers.items():
            q = p[p["model"] == model]
            if q.empty:
                continue

            ax.scatter(
                q["throughput"],
                q["energy_per_item_j"],
                marker=marker,
                s=30,
                alpha=0.80,
                label=model,
                color=COLORS[device],
                edgecolor="white",
                linewidth=0.35,
            )

        boundaries = [
            str(x)
            for x in p["energy_boundary"].dropna().unique()
        ]
        boundary = (
            boundaries[0]
            if len(boundaries) == 1
            else "/".join(boundaries)
            if boundaries
            else "unknown boundary"
        )

        ax.set_title(f"{device}\n({BOUNDARY_LABELS.get(boundary, boundary)})")
        ax.set_xlabel("Throughput (images/s)")
        ax.set_ylabel("Energy (J/image)")
        ax.grid(alpha=0.22)

    handles, labels = axes[0].get_legend_handles_labels()

    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.06),
            ncol=3,
            frameon=False,
        )

    fig.text(
        0.5,
        -0.035,
        "Within each panel, desirable configurations move "
        "toward the lower-right region.",
        ha="center",
        fontsize=7,
        color="0.35",
    )

    fig.subplots_adjust(
        top=0.79,
        bottom=0.22,
        wspace=0.42
    )

    save(fig, "fig6_speed_energy_tradeoff")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print(f"Input rows: {len(df)}")
    print(f"Output: {OUT}")

    build_crossover()
    build_thread_scaling()
    build_precision_throughput()
    build_memory()
    build_power()
    build_energy_tradeoff()

    print("\nDone.")
    print("Generated paper figures:")
    for p in sorted(OUT.glob("fig*.pdf")):
        print(" -", p.relative_to(ROOT))

    print("\nFigure source data:")
    for p in sorted(DATA_OUT.glob("fig*.csv")):
        print(" -", p.relative_to(ROOT))
