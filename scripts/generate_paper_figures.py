#!/usr/bin/env python3
"""Generate publication figures from benchmark summary JSON files in results/."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results")
FIG_DIR = Path("paper/figures")


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "figure.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_fig1_data_efficiency() -> None:
    """Figure 1: Cross-subject data-efficiency scaling on ds005107 (Age & Visual Detection)."""
    age = json.loads((RESULTS_DIR / "summary_aligned21_age.json").read_text())
    det = json.loads((RESULTS_DIR / "summary_aligned21_visual_detection.json").read_text())

    budgets = [1, 2, 4, 10, 20]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)

    arm_specs = [
        ("pretrained_linear_probe", "BrainOmni Pre-trained (MAG=1)", "#1f77b4", "o", "-"),
        ("phase5_opm_aware_probe", "OPM-Aware Adapter (GRAD=2)", "#2ca02c", "s", "-"),
        ("control_b_linear_probe", "Control B (Tokenizer Only)", "#9467bd", "^", "--"),
        ("control_a_linear_probe", "Control A (Random Init)", "#7f7f7f", "d", ":"),
        ("compact_megnet", "CompactMEGNet (1.5k CNN)", "#d62728", "v", "-."),
        ("classical_bandpower", "Classical Band-Power", "#ff7f0e", "P", "--"),
    ]

    # Panel A: High-SNR Evoked Visual Detection
    ax = axes[0]
    for arm_key, label, color, marker, ls in arm_specs:
        if arm_key not in det["results_by_budget"]["1"]:
            continue
        means = [det["results_by_budget"][str(b)][arm_key]["mean"] * 100 for b in budgets]
        lows = [det["results_by_budget"][str(b)][arm_key]["ci_low"] * 100 for b in budgets]
        highs = [det["results_by_budget"][str(b)][arm_key]["ci_high"] * 100 for b in budgets]
        ax.plot(budgets, means, label=label, color=color, marker=marker, linestyle=ls, lw=1.8)
        ax.fill_between(budgets, lows, highs, color=color, alpha=0.12)
    ax.set_title("A. Evoked Visual Detection (ds005107, N=21)")
    ax.set_xlabel("Training Subject Budget (B)")
    ax.set_ylabel("Balanced Accuracy (%)")
    ax.set_xticks(budgets)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", frameon=True)

    # Panel B: Cross-Subject Demographic Decoding (Age)
    ax2 = axes[1]
    for arm_key, label, color, marker, ls in arm_specs:
        if arm_key not in age["results_by_budget"]["1"]:
            continue
        means = [age["results_by_budget"][str(b)][arm_key]["mean"] * 100 for b in budgets]
        lows = [age["results_by_budget"][str(b)][arm_key]["ci_low"] * 100 for b in budgets]
        highs = [age["results_by_budget"][str(b)][arm_key]["ci_high"] * 100 for b in budgets]
        ax2.plot(budgets, means, label=label, color=color, marker=marker, linestyle=ls, lw=1.8)
        ax2.fill_between(budgets, lows, highs, color=color, alpha=0.12)
    ax2.axhline(50.0, color="black", linestyle="--", lw=1.0, alpha=0.6, label="Chance (50%)")
    ax2.set_title("B. Cross-Subject Face Attribute Decoding (Age, N=21)")
    ax2.set_xlabel("Training Subject Budget (B)")
    ax2.set_ylabel("Balanced Accuracy (%)")
    ax2.set_xticks(budgets)
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig1_data_efficiency_curves.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_fig2_sensor_ablations() -> None:
    """Figure 2: Sensor-type embedding asymmetry and geometry ablations on ds005107 (N=21)."""
    ab = json.loads((RESULTS_DIR / "summary_phase3_ablations.json").read_text())
    pcc_dict = ab["zero_shot_tokenizer_pcc_by_geometry_mode"]

    modes = [
        ("opm_as_grad", "Remapped\n(GRAD=2)", "#2ca02c"),
        ("default", "Default\n(MAG=1)", "#1f77b4"),
        ("zero_orientation", "Zero\nOrientation", "#ff7f0e"),
        ("flip_orientation", "Inverted\nOrientation", "#d62728"),
        ("scramble_channels", "Scrambled\nTopology", "#7f7f7f"),
    ]

    labels = [m[1] for m in modes]
    colors = [m[2] for m in modes]
    means = [pcc_dict[m[0]]["mean_pcc"] for m in modes]
    err_low = [pcc_dict[m[0]]["mean_pcc"] - pcc_dict[m[0]]["ci_low"] for m in modes]
    err_high = [pcc_dict[m[0]]["ci_high"] - pcc_dict[m[0]]["mean_pcc"] for m in modes]

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    x = np.arange(len(modes))
    bars = ax.bar(
        x,
        means,
        yerr=[err_low, err_high],
        capsize=5,
        color=colors,
        alpha=0.88,
        edgecolor="black",
        linewidth=0.8,
        width=0.58,
    )
    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.012,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 0.44)
    ax.set_ylabel("Zero-Shot Waveform Reconstruction (PCC)")
    ax.set_title("Sensor-Type Embedding Calibration & Geometry Perturbations (ds005107, N=21)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig2_sensor_geometry_ablations.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_fig3_nottingham_triaxial() -> None:
    """Figure 3: Triaxial spatial bottleneck and axis-partitioned adapter on Nottingham (N=10)."""
    nott = json.loads((RESULTS_DIR / "summary_phase4_nottingham.json").read_text())
    cs = nott["cohort_bootstrap_summary"]

    modes = [
        ("triaxial_xyz_raw", "Monolithic\nTriaxial (X/Y/Z)", "#7f7f7f"),
        ("triaxial_xyz_softhfc", "Triaxial +\nSoft-HFC", "#1f77b4"),
        ("tangential_only_xy", "Tangential\nOnly (X/Y)", "#ff7f0e"),
        ("phase5_axis_split_xyz", "Axis-Partitioned\nAdapter (X/Y/Z)", "#2ca02c"),
        ("radial_only_z", "Radial Isolation\n(Z-Axis Only)", "#9467bd"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    x = np.arange(len(modes))
    labels = [m[1] for m in modes]
    colors = [m[2] for m in modes]

    # Panel A: Zero-shot PCC
    pcc_means = [cs[m[0]]["zero_shot_pcc"]["mean"] for m in modes]
    pcc_low = [cs[m[0]]["zero_shot_pcc"]["mean"] - cs[m[0]]["zero_shot_pcc"]["ci_low"] for m in modes]
    pcc_high = [cs[m[0]]["zero_shot_pcc"]["ci_high"] - cs[m[0]]["zero_shot_pcc"]["mean"] for m in modes]

    bars1 = axes[0].bar(
        x, pcc_means, yerr=[pcc_low, pcc_high], capsize=4, color=colors, alpha=0.88, edgecolor="black", width=0.58
    )
    for bar, val in zip(bars1, pcc_means):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, val + 0.038, f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold"
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylim(0.0, 0.62)
    axes[0].set_ylabel("Zero-Shot Reconstruction PCC")
    axes[0].set_title("A. Triaxial Waveform Reconstruction (Zenodo 11126593, N=10)")
    axes[0].grid(True, axis="y", linestyle=":", alpha=0.5)

    # Panel B: Tactile Braille Decoding BACC
    bacc_means = [cs[m[0]]["braille_5fold_bacc"]["pretrained_linear_probe"]["mean"] * 100 for m in modes]
    bacc_low = [
        bacc_means[i] - cs[m[0]]["braille_5fold_bacc"]["pretrained_linear_probe"]["ci_low"] * 100
        for i, m in enumerate(modes)
    ]
    bacc_high = [
        cs[m[0]]["braille_5fold_bacc"]["pretrained_linear_probe"]["ci_high"] * 100 - bacc_means[i]
        for i, m in enumerate(modes)
    ]
    bars2 = axes[1].bar(
        x, bacc_means, yerr=[bacc_low, bacc_high], capsize=4, color=colors, alpha=0.88, edgecolor="black", width=0.58
    )
    for bar, val in zip(bars2, bacc_means):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, val + 7.2, f"{val:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold"
        )
    axes[1].axhline(50.0, color="black", linestyle="--", lw=1.0, alpha=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylim(35.0, 74.0)
    axes[1].set_ylabel("5-Fold Tactile Braille BACC (%)")
    axes[1].set_title("B. Somatosensory Finger Decoding (Start_index vs Start_pinky)")
    axes[1].grid(True, axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"fig3_nottingham_triaxial_bottleneck.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _configure_style()
    plot_fig1_data_efficiency()
    plot_fig2_sensor_ablations()
    plot_fig3_nottingham_triaxial()
    print(f"Generated publication figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
