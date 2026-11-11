"""Rebuild the short-paper figure from saved per-participant results.

Every plotted value is read from results/*.json; no model inference runs here.
Panels that compare conditions use per-participant records so that
within-participant consistency, not only the cohort mean, is visible.

The script also writes figures/transfer_overview_values.json, a sidecar listing
every number the figure displays. audit/verify.py recomputes those numbers from
the saved results and asserts they agree, so values carried by the figure rather
than by the manuscript text remain under verification.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from factorial_results import write as write_factorial

BASE = Path(__file__).resolve().parent
RESULTS = BASE.parents[1] / "results"
OUT = BASE / "figures"
OUT.mkdir(exist_ok=True)

INK = "#0A1F1C"
SAGE = "#3D5A52"
ACCENT = "#00695C"
MUTED = "#9AA7A2"
GRID = "#DCDCDC"
BAND = "#F2F4F3"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.0,
        "axes.labelsize": 7.0,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "savefig.facecolor": "white",
    }
)


def bootstrap_ci(values: np.ndarray, n_boot: int = 20000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of the mean over participants."""
    rng = np.random.default_rng(seed)
    n = len(values)
    means = values[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def exact_sign_flip_p(delta: np.ndarray) -> float:
    """Exact two-sided sign-flip permutation p-value (feasible for n <= 21)."""
    n = len(delta)
    observed = abs(delta.mean())
    hits = sum(
        1
        for signs in itertools.product((1, -1), repeat=n)
        if abs((delta * np.array(signs)).mean()) >= observed - 1e-15
    )
    return hits / 2**n


def panel_label(ax, letter: str, text: str) -> None:
    ax.set_title(f"{letter}  {text}", loc="left", fontweight="bold", pad=4, color=INK)


def slopegraph(ax, arrays: Sequence[np.ndarray], labels: Sequence[str], ylabel: str) -> None:
    """Paired per-participant trajectories across two or more conditions."""
    xs = list(range(len(arrays)))
    stacked = np.vstack(arrays)
    for col in stacked.T:
        ax.plot(xs, col, color=MUTED, linewidth=0.6, alpha=0.9, zorder=1)
    means = stacked.mean(axis=1)
    ax.plot(xs, means, color=ACCENT, linewidth=1.8, zorder=3)
    ax.scatter(xs, means, color=ACCENT, s=16, zorder=4)
    ax.set_xticks(xs, labels)
    ax.set_xlim(-0.35, len(arrays) - 1 + 0.35)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def contrast_panel(ax, rows: Sequence[tuple[str, np.ndarray]], xlabel: str) -> list[dict]:
    """Paired contrasts with participant-bootstrap confidence intervals.

    The consistency count is placed in the row label rather than inside the
    axes, so it cannot collide with a marker near the panel edge.
    """
    labels, records = [], []
    for i, (label, delta) in enumerate(rows):
        lo, hi = bootstrap_ci(delta)
        mean = float(delta.mean())
        direction = 1 if mean >= 0 else -1
        n_consistent = int((delta * direction > 0).sum())
        ax.errorbar(
            mean, i, xerr=[[mean - lo], [hi - mean]], fmt="o",
            color=ACCENT if mean > 0 else SAGE,
            markersize=4, capsize=2.5, linewidth=1.2,
        )
        labels.append(f"{label}\n{n_consistent}/{len(delta)}")
        records.append(
            {"row": label, "mean": mean, "ci_low": lo, "ci_high": hi,
             "consistent": n_consistent, "direction": "positive" if direction > 0 else "negative", "n": len(delta)}
        )
    ax.axvline(0.0, color=INK, linewidth=0.7)
    ax.set_yticks(range(len(rows)), labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    return records


def dumbbell_panel(
    ax,
    rows: Sequence[tuple[str, np.ndarray, np.ndarray]],
    xlabel: str,
    rule_after: int | None = None,
) -> list[dict]:
    """Two paired arms per row, drawn as connected markers with bootstrap CIs.

    Row labels, consistency counts and the legend all sit outside the data area,
    so nothing competes with the markers. Alternating bands keep the rows
    readable without gridlines crossing the intervals.
    """
    labels, records = [], []
    for i, (label, probe, classical) in enumerate(rows):
        if i % 2 == 1:
            ax.axhspan(i - 0.5, i + 0.5, color=BAND, zorder=0)
        p_mean, c_mean = float(probe.mean()), float(classical.mean())
        p_lo, p_hi = bootstrap_ci(probe)
        c_lo, c_hi = bootstrap_ci(classical)
        n_pos = int((classical > probe).sum())
        ax.plot([p_mean, c_mean], [i, i], color=SAGE, linewidth=1.6, alpha=0.55, zorder=2)
        ax.errorbar(
            p_mean, i, xerr=[[p_mean - p_lo], [p_hi - p_mean]], fmt="o",
            markerfacecolor="white", markeredgecolor=SAGE, ecolor=SAGE,
            markersize=5.0, markeredgewidth=1.1, capsize=0, linewidth=0.8,
            alpha=0.95, zorder=3,
        )
        ax.errorbar(
            c_mean, i, xerr=[[c_mean - c_lo], [c_hi - c_mean]], fmt="o",
            color=ACCENT, markersize=5.0, capsize=0, linewidth=0.8, zorder=4,
        )
        labels.append(label)
        ax.text(
            1.025, i, f"{n_pos}/{len(probe)}", transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=6, color=SAGE,
        )
        records.append(
            {"row": label, "probe": p_mean, "probe_ci": [p_lo, p_hi],
             "classical": c_mean, "classical_ci": [c_lo, c_hi],
             "consistent": n_pos, "n": len(probe)}
        )
    if rule_after is not None:
        ax.axhline(rule_after + 0.5, color=INK, linewidth=0.7, alpha=0.75, zorder=6)
    ax.axvline(50.0, color=INK, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    ax.set_yticks(range(len(rows)), labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    return records


# ---------------------------------------------------------------------------
# Saved results
# ---------------------------------------------------------------------------
single = json.loads((RESULTS / "summary_phase3_ablations.json").read_text())
single = single["zero_shot_tokenizer_pcc_by_geometry_mode"]
per_subject = {k: np.asarray(v["per_subject_pcc"], dtype=float) for k, v in single.items()}

tri = json.loads((RESULTS / "summary_phase4_nottingham_factorial_and_seeds.json").read_text())
subjects = sorted(tri["per_subject"])


def bacc(tag: str, mode: str, arm: str) -> np.ndarray:
    blocks = [tri["per_subject"][s]["softhfc"][tag] for s in subjects]
    return 100.0 * np.asarray([(b["random_partitions"]["42"] if mode == "random" else b[mode])["braille_5fold_bacc"][arm] for b in blocks])


squid = json.loads((RESULTS / "summary_ds000117_squid_baseline.json").read_text())
squid = squid["reconstruction_pcc"]

# ---------------------------------------------------------------------------
# Figure: two rows, sensor metadata above, triaxial channel organisation below
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(
    2, 3, figsize=(7.0, 3.70), gridspec_kw={"width_ratios": [0.78, 1.18, 1.10]}
)

# A: paired sensor-tag trajectories on FACE-DEC
slopegraph(
    axes[0, 0], [per_subject["default"], per_subject["opm_as_grad"]],
    ["MAG", "GRAD"], "Reconstruction PCC",
)
panel_label(axes[0, 0], "A", "Sensor tag")

# B: all single-axis metadata interventions against the default tag
single_rows = contrast_panel(
    axes[0, 1],
    [
        ("Gradiometer tag", per_subject["opm_as_grad"] - per_subject["default"]),
        ("Zero orientation", per_subject["zero_orientation"] - per_subject["default"]),
        ("Reversed orientation", per_subject["flip_orientation"] - per_subject["default"]),
        ("Scrambled geometry", per_subject["scramble_channels"] - per_subject["default"]),
    ],
    r"$\Delta$ PCC vs default tag",
)
panel_label(axes[0, 1], "B", "Single-axis OPM (N=21)")
axes[0, 1].set_xlim(-0.045, 0.165)

# C: single-participant SQUID control, where the same relabelling applies
squid_bars = [
    ("MAG\n(102 ch)", squid["mag_102ch"], SAGE),
    ("GRAD\n(204 ch)", squid["grad_204ch"], SAGE),
    ("MAG as\nGRAD", squid["mag_as_grad"], ACCENT),
]
axes[0, 2].bar(
    range(3), [b[1] for b in squid_bars], color=[b[2] for b in squid_bars], width=0.6
)
for i, (_, value, _) in enumerate(squid_bars):
    axes[0, 2].text(i, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=6.5)
axes[0, 2].set_xticks(range(3), [b[0] for b in squid_bars])
axes[0, 2].set_ylim(0, 0.84)
axes[0, 2].set_ylabel("Reconstruction PCC")
axes[0, 2].grid(axis="y", color=GRID, linewidth=0.5)
axes[0, 2].set_axisbelow(True)
panel_label(axes[0, 2], "C", "SQUID control (1 participant)")

factorial_report, fx, factorial_contrasts = write_factorial()

# D: paired monolithic vs axis-partitioned tokenisation, matched filtering
slopegraph(
    axes[1, 0],
    [fx["softhfc_mag_monolithic"], fx["softhfc_mag_random"], fx["softhfc_mag_axis_partitioned"]],
    ["Mono", "Rand", "Axis"], "Reconstruction PCC",
)
panel_label(axes[1, 0], "D", "MAG, Soft-HFC")

# E: triaxial contrasts, each against a baseline with identical filtering
tri_rows = contrast_panel(
    axes[1, 1],
    [
        ("MAG, Soft-HFC", factorial_contrasts["softhfc_mag_grouping"]),
        ("MAG, raw", factorial_contrasts["raw_mag_grouping"]),
        ("GRAD, Soft-HFC", factorial_contrasts["softhfc_grad_grouping"]),
        ("GRAD, raw", factorial_contrasts["raw_grad_grouping"]),
    ],
    r"Axis minus random, $\Delta$ PCC",
)
panel_label(axes[1, 1], "E", "Triaxial OPM (N=10)")
axes[1, 1].set_xlim(-0.012, 0.155)

# F: the decoding endpoint, probe against a classical baseline on the same trials
DEC_CONFIGS = [(f"{tag.upper()} {label}", tag, mode)
               for tag in ("mag", "grad")
               for mode, label in (("monolithic", "mono"), ("random", "rand"), ("axis_partitioned", "axis"))]
dec_rows = dumbbell_panel(
    axes[1, 2],
    [(label, bacc(tag, mode, "pretrained_linear_probe"), bacc(tag, mode, "classical_bandpower"))
     for label, tag, mode in DEC_CONFIGS],
    "Braille balanced accuracy (%)",
    rule_after=2,
)
panel_label(axes[1, 2], "F", "Decoding, Soft-HFC")
axes[1, 2].set_xlim(39, 59)
axes[1, 2].set_ylim(5.6, -1.0)
legend_handles = [
    plt.Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
               markeredgecolor=SAGE, markeredgewidth=1.0, markersize=4.5,
               label="Pretrained probe"),
    plt.Line2D([], [], marker="o", linestyle="none", color=ACCENT, markersize=4.5,
               label="Classical baseline"),
]
fig.legend(
    handles=legend_handles, loc="lower right", bbox_to_anchor=(0.99, 0.005),
    bbox_transform=fig.transFigure,
    ncol=2, fontsize=6, frameon=False, handletextpad=0.3, borderpad=0.0,
    columnspacing=1.2,
)

# Reserve a separate footer for the legend, outside all axes and labels.
fig.tight_layout(pad=0.25, w_pad=1.5, h_pad=1.25, rect=(0, 0.105, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"transfer_overview.{ext}", dpi=400, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# ---------------------------------------------------------------------------
# Sidecar: every value the figure displays, for audit/verify.py
# ---------------------------------------------------------------------------
sidecar = {
    "single_axis_contrasts": single_rows,
    "triaxial_contrasts": tri_rows,
    "decoding": dec_rows,
    "squid_control": {"mag_102ch": squid["mag_102ch"], "grad_204ch": squid["grad_204ch"],
                      "mag_as_grad": squid["mag_as_grad"]},
    "slopegraph_means": {
        "single_axis_default": float(per_subject["default"].mean()),
        "single_axis_opm_as_grad": float(per_subject["opm_as_grad"].mean()),
        "triaxial_monolithic_softhfc": float(fx["softhfc_mag_monolithic"].mean()),
        "triaxial_random_softhfc": float(fx["softhfc_mag_random"].mean()),
        "triaxial_axis_split": float(fx["softhfc_mag_axis_partitioned"].mean()),
    },
}
(OUT / "transfer_overview_values.json").write_text(json.dumps(sidecar, indent=2) + "\n")

# ---------------------------------------------------------------------------
# Console report: the contrasts quoted in the manuscript text
# ---------------------------------------------------------------------------
report = {"single_axis_remap": per_subject["opm_as_grad"] - per_subject["default"], **factorial_contrasts}
for name, delta in report.items():
    lo, hi = bootstrap_ci(delta)
    print(
        f"{name}: delta={delta.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] "
        f"consistent={int((delta * np.sign(delta.mean()) > 0).sum())}/{len(delta)} p_exact={exact_sign_flip_p(delta):.2e}"
    )
for row in dec_rows:
    print(
        f"decoding {row['row']}: probe={row['probe']:.2f} classical={row['classical']:.2f} "
        f"classical_leads={row['consistent']}/{row['n']}"
    )
print("Wrote figures/transfer_overview.{pdf,png} and transfer_overview_values.json.")
