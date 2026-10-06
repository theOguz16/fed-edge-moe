from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG_DATA = ROOT / "paper" / "figures" / "data"
OUT = ROOT / "paper" / "tables"

OUT.mkdir(parents=True, exist_ok=True)

UNIFIED = RESULTS / "unified_characterization.csv"
CROSSOVER = FIG_DATA / "fig1_device_crossover.csv"

if not UNIFIED.exists():
    raise FileNotFoundError(UNIFIED)

if not CROSSOVER.exists():
    raise FileNotFoundError(CROSSOVER)


# ============================================================
# Helpers
# ============================================================

def write_tex(filename, content):
    path = OUT / filename
    path.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"[OK] {path.relative_to(ROOT)}")


def latex_escape(text):
    text = str(text)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


df = pd.read_csv(UNIFIED)


# ============================================================
# TABLE I — Hardware
# ============================================================

hardware_tex = r"""
\begin{table}[t]
\caption{Experimental Hardware Platforms}
\label{tab:hardware}
\centering
\footnotesize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llll}
\hline
\textbf{Platform} &
\textbf{CPU} &
\textbf{Accelerator} &
\textbf{Memory Architecture} \\
\hline
Apple M4 system &
Apple M4, 10-core CPU &
Integrated Apple GPU (MPS) &
Unified memory \\
MSI laptop &
Intel Core i7-11800H, 8C/16T &
NVIDIA RTX 3050 Laptop, 4 GB &
System RAM + discrete VRAM \\
\hline
\end{tabular}%
}
\end{table}
"""

write_tex("table1_hardware.tex", hardware_tex)


# ============================================================
# TABLE II — Workload Matrix
# ============================================================

workload = (
    df[["modality", "difficulty", "model"]]
    .drop_duplicates()
    .copy()
)

difficulty_order = ["easy", "medium", "hard"]
modality_order = ["text", "vision"]

matrix = {}

for modality in modality_order:
    matrix[modality] = {}

    for difficulty in difficulty_order:
        rows = workload[
            (workload["modality"] == modality)
            & (workload["difficulty"] == difficulty)
        ]

        if rows.empty:
            matrix[modality][difficulty] = "--"
        else:
            model_name = rows.iloc[0]["model"]

            if model_name == "Qwen2.5-1.5B-Instruct":
                model_name = "Qwen2.5-1.5B"

            matrix[modality][difficulty] = latex_escape(
                model_name
            )


workload_tex = rf"""
\begin{{table}}[t]
\caption{{Evaluated Workload Matrix}}
\label{{tab:workloads}}
\centering
\footnotesize
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{llll}}
\hline
\textbf{{Modality}} &
\textbf{{Easy}} &
\textbf{{Medium}} &
\textbf{{Hard}} \\
\hline
Text &
{matrix["text"]["easy"]} &
{matrix["text"]["medium"]} &
{matrix["text"]["hard"]} \\
Vision &
{matrix["vision"]["easy"]} &
{matrix["vision"]["medium"]} &
{matrix["vision"]["hard"]} \\
\hline
\end{{tabular}}%
}}
\end{{table}}
"""

write_tex("table2_workloads.tex", workload_tex)


# ============================================================
# TABLE III — Representative crossover results
# ============================================================

cross = pd.read_csv(CROSSOVER)

meta = (
    df[["model", "modality", "difficulty"]]
    .drop_duplicates()
)

cross = cross.merge(meta, on="model", how="left")

modality_rank = {
    "text": 0,
    "vision": 1,
}

difficulty_rank = {
    "easy": 0,
    "medium": 1,
    "hard": 2,
}

cross["_modality_order"] = cross["modality"].map(modality_rank)
cross["_difficulty_order"] = cross["difficulty"].map(difficulty_rank)

cross = cross.sort_values(
    ["_modality_order", "_difficulty_order"]
)

display_names = {
    "Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B",
}

rows = []

for _, row in cross.iterrows():
    model = display_names.get(row["model"], row["model"])

    rows.append(
        "{} & {} & {} & {:.2f}$\\times$ \\\\".format(
            latex_escape(model),
            str(row["modality"]).capitalize(),
            str(row["difficulty"]).capitalize(),
            row["median_msi_over_mac"],
        )
    )

result_rows = "\n".join(rows)

results_tex = rf"""
\begin{{table}}[t]
\caption{{Cross-Device Throughput Summary for Matched Short Runs}}
\label{{tab:crossover}}
\centering
\footnotesize
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{llll}}
\hline
\textbf{{Model}} &
\textbf{{Modality}} &
\textbf{{Difficulty}} &
\textbf{{Median MSI/M4 Ratio}} \\
\hline
{result_rows}
\hline
\end{{tabular}}%
}}
\vspace{{1mm}}

\scriptsize
A ratio greater than 1 indicates higher throughput on the MSI-side platform for the matched configuration.
\end{{table}}
"""

write_tex("table3_representative_results.tex", results_tex)


print("\nGenerated paper tables:")

for path in sorted(OUT.glob("table*.tex")):
    print(" -", path.relative_to(ROOT))
