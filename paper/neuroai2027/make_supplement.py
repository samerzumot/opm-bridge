"""Generate scientific supplement tables from saved participant results."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
RESULTS = ROOT / "results"
OUT = BASE / "supplement_tables.tex"

SINGLE_MODES = [
    ("default", r"Default (\texttt{MAG=1})"),
    ("opm_as_grad", r"Gradiometer tag (\texttt{GRAD=2})"),
    ("zero_orientation", "Zero orientation"),
    ("flip_orientation", "Reversed orientation"),
    ("scramble_channels", "Scrambled geometry"),
]
TRI_MODES = [
    ("triaxial_xyz_raw", "Monolithic", "no"),
    ("triaxial_xyz_softhfc", "Monolithic", "Soft"),
    ("random_3way_split", "Random 3-way", "no"),
    ("random_3way_split_softhfc", "Random 3-way", "Soft"),
    ("axis_split_raw", "Axis partition", "no"),
    ("phase5_axis_split_xyz", "Axis partition", "Soft"),
    ("radial_only_z", r"Radial only ($Z$)", "no"),
    ("tangential_only_xy", r"Tangential only ($X,Y$)", "no"),
]
DEC_ARMS = [
    ("pretrained_linear_probe", "Pretrained probe"),
    ("control_a_linear_probe", "Random init (Control A)"),
    ("classical_bandpower", "Classical baseline"),
]
def bootstrap_ci(v: np.ndarray, n_boot: int = 20000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def exact_sign_flip_p(delta: np.ndarray) -> float:
    observed = abs(delta.mean())
    hits = sum(
        1
        for signs in itertools.product((1, -1), repeat=len(delta))
        if abs((delta * np.array(signs)).mean()) >= observed - 1e-15
    )
    return hits / 2 ** len(delta)


def contrast_row(label: str, delta: np.ndarray, positive: bool) -> str:
    lo, hi = bootstrap_ci(delta)
    n_dir = int((delta > 0).sum() if positive else (delta < 0).sum())
    return (
        f"{label} & ${delta.mean():+.4f}$ & $[{lo:+.4f}, {hi:+.4f}]$ & "
        f"{n_dir}/{len(delta)} & {exact_sign_flip_p(delta):.2e} \\\\"
    )


single = json.loads((RESULTS / "summary_phase3_ablations.json").read_text())
single = single["zero_shot_tokenizer_pcc_by_geometry_mode"]
sa = {k: np.asarray(v["per_subject_pcc"], dtype=float) for k, v in single.items()}

tri = json.loads((RESULTS / "summary_phase4_nottingham_full.json").read_text())
subjects = sorted(tri["per_subject"])
pcc = {m: np.asarray([tri["per_subject"][s][m]["zero_shot_tokenizer_pcc"] for s in subjects])
       for m, _, _ in TRI_MODES}


def bacc(mode: str, arm: str) -> np.ndarray:
    return 100.0 * np.asarray(
        [tri["per_subject"][s][mode]["braille_5fold_bacc"][arm] for s in subjects]
    )


L: list[str] = []
A = L.append

# --- single-axis per-participant -------------------------------------------
A(r"\begin{table*}[htbp]\centering\footnotesize")
A(r"\setlength{\tabcolsep}{2.5pt}")
A(r"\begin{tabular}{l" + "r" * len(SINGLE_MODES) + "}")
A(r"\toprule")
A("Participant & " + " & ".join(
    ["Default", "GRAD tag", "Zero ori.", "Rev. ori.", "Scrambled"]) + r" \\")
A(r"\midrule")
for i in range(len(sa["default"])):
    A(f"P{i + 1:02d} & " + " & ".join(f"{sa[m][i]:.4f}" for m, _ in SINGLE_MODES) + r" \\")
A(r"\midrule")
A("Mean & " + " & ".join(f"{sa[m].mean():.4f}" for m, _ in SINGLE_MODES) + r" \\")
A(r"\bottomrule\end{tabular}")
A(r"\caption{Zero-shot reconstruction PCC per participant on FACE-DEC (ds005107), "
  r"50 shared channels, Soft-HFC $\alpha=0.25$. Participants are indexed in the saved "
  r"record order, which is not the dataset's own participant numbering. Waveforms are "
  r"identical across columns; only the metadata supplied to the tokenizer differs.}")
A(r"\label{tab:s-single-per-subject}\end{table*}")
A("")

# --- single-axis contrasts --------------------------------------------------
A(r"\begin{table*}[htbp]\centering\small")
A(r"\begin{tabular}{lrrrr}")
A(r"\toprule")
A(r"Contrast vs default & Mean $\Delta$ & 95\% CI & Consistent & Exact $p$ \\")
A(r"\midrule")
A(contrast_row("Gradiometer tag", sa["opm_as_grad"] - sa["default"], True))
for mode, label in SINGLE_MODES[2:]:
    A(contrast_row(label, sa[mode] - sa["default"], False))
A(r"\bottomrule\end{tabular}")
A(r"\caption{Paired single-axis contrasts, 21 participants. Confidence intervals are "
  r"percentile bootstrap over participants, 20{,}000 resamples, seed 0. The consistency "
  r"column counts participants moving in the stated direction. $p$ is the exact "
  r"two-sided sign-flip permutation value enumerating all $2^{21}$ assignments.}")
A(r"\label{tab:s-single-contrasts}\end{table*}")
A("")

# --- triaxial grid, cohort --------------------------------------------------
A(r"\begin{table*}[htbp]\centering\small")
A(r"\begin{tabular}{llrr}")
A(r"\toprule")
A(r"Configuration & HFC & Mean PCC & 95\% CI \\")
A(r"\midrule")
for i, (mode, label, hfc) in enumerate(TRI_MODES):
    if i == 6:
        A(r"\midrule")
    lo, hi = bootstrap_ci(pcc[mode])
    A(f"{label} & {hfc} & {pcc[mode].mean():.4f} & $[{lo:.4f}, {hi:.4f}]$ \\\\")
A(r"\bottomrule\end{tabular}")
A(r"\caption{Triaxial reconstruction, 10 adult participants, 84 trials each. The upper "
  r"block is the complete $2\times3$ grid crossing partition type with filtering. The "
  r"lower block reconstructs channel subsets and is not comparable to the full-array "
  r"rows. Confidence intervals use participant bootstrap with 20{,}000 resamples "
  r"and seed 0.}")
A(r"\label{tab:s-tri-cohort}\end{table*}")
A("")

# --- triaxial per-participant ----------------------------------------------
A(r"\begin{table*}[htbp]\centering\small")
A(r"\setlength{\tabcolsep}{4pt}")
A(r"\begin{tabular}{l" + "r" * len(TRI_MODES) + "}")
A(r"\toprule")
A(r"Participant & Mono. raw & Mono. Soft & Rand. raw & Rand. Soft & Axis raw & "
  r"Axis Soft & Radial & Tangential \\")
A(r"\midrule")
for i, s in enumerate(subjects):
    A(f"P{i + 1:02d} & " + " & ".join(f"{pcc[m][i]:.4f}" for m, _, _ in TRI_MODES) + r" \\")
A(r"\midrule")
A("Mean & " + " & ".join(f"{pcc[m].mean():.4f}" for m, _, _ in TRI_MODES) + r" \\")
A(r"\bottomrule\end{tabular}")
A(r"\caption{Zero-shot reconstruction PCC per participant across the triaxial grid "
  r"(Zenodo 11126593, the 10 adult participants). Every contrast in the main text is "
  r"computed from these columns within a single filtering regime.}")
A(r"\label{tab:s-tri-per-subject}\end{table*}")
A("")

# --- all matched triaxial contrasts ----------------------------------------
A(r"\begin{table*}[htbp]\centering\small")
A(r"\begin{tabular}{lrrrr}")
A(r"\toprule")
A(r"Contrast (matched filtering) & Mean $\Delta$ & 95\% CI & Consistent & Exact $p$ \\")
A(r"\midrule")
A(r"\multicolumn{5}{l}{\textit{Grouping: axis partition against a size-matched random partition}} \\")
A(contrast_row("Soft-HFC", pcc["phase5_axis_split_xyz"] - pcc["random_3way_split_softhfc"], True))
A(contrast_row("Raw", pcc["axis_split_raw"] - pcc["random_3way_split"], True))
A(r"\midrule")
A(r"\multicolumn{5}{l}{\textit{Capacity: random partition against the monolithic tokenizer}} \\")
A(contrast_row("Soft-HFC", pcc["random_3way_split_softhfc"] - pcc["triaxial_xyz_softhfc"], True))
A(contrast_row("Raw", pcc["random_3way_split"] - pcc["triaxial_xyz_raw"], True))
A(r"\midrule")
A(r"\multicolumn{5}{l}{\textit{Total: axis partition against the monolithic tokenizer}} \\")
A(contrast_row("Soft-HFC", pcc["phase5_axis_split_xyz"] - pcc["triaxial_xyz_softhfc"], True))
A(contrast_row("Raw", pcc["axis_split_raw"] - pcc["triaxial_xyz_raw"], True))
A(r"\midrule")
A(r"\multicolumn{5}{l}{\textit{Filtering: Soft-HFC applied within each partition type}} \\")
A(contrast_row("Monolithic", pcc["triaxial_xyz_softhfc"] - pcc["triaxial_xyz_raw"], False))
A(contrast_row("Random 3-way", pcc["random_3way_split_softhfc"] - pcc["random_3way_split"], False))
A(contrast_row("Axis partition", pcc["phase5_axis_split_xyz"] - pcc["axis_split_raw"], False))
A(r"\bottomrule\end{tabular}")
A(r"\caption{All paired triaxial contrasts, 10 participants. Every row compares two "
  r"conditions with identical filtering, or one partition type under both filtering "
  r"settings. With $n=10$ the smallest attainable exact $p$ is $2/2^{10}\approx0.002$, "
  r"so values at that floor indicate perfect sign consistency rather than a resolved tail.}")
A(r"\label{tab:s-tri-contrasts}\end{table*}")
A("")

# --- decoding ---------------------------------------------------------------
A(r"\begin{table*}[htbp]\centering\small")
A(r"\setlength{\tabcolsep}{4pt}")
A(r"\begin{tabular}{llrrrrr}")
A(r"\toprule")
A(r"Configuration & HFC & Probe & Ctrl A & Classical & Classical 95\% CI & Cl. $>$ Pr. \\")
A(r"\midrule")
for mode, label, hfc in TRI_MODES:
    vals = [bacc(mode, arm) for arm, _ in DEC_ARMS]
    lo, hi = bootstrap_ci(vals[2])
    lead = int((vals[2] > vals[0]).sum())
    A(f"{label} & {hfc} & " + " & ".join(f"{v.mean():.2f}" for v in vals)
      + f" & $[{lo:.2f}, {hi:.2f}]$ & {lead}/{len(subjects)} \\\\")
A(r"\bottomrule\end{tabular}")
A(r"\caption{Braille decoding balanced accuracy (\%), five-fold within participant, "
  r"10 adult participants. The classical arm computes six log band powers spanning "
  r"1--96\,Hz plus four temporal window means per channel, so it does not depend on how "
  r"channels are grouped for the tokenizer and takes the same value across partitions "
  r"sharing a channel set and filtering. Its interval excludes chance only in the "
  r"Soft-HFC full-array conditions. The final column counts participants in which the "
  r"classical decoder exceeds the pretrained probe.}")
A(r"\label{tab:s-decoding}\end{table*}")
A("")

OUT.write_text("\n".join(L) + "\n")
print(f"wrote {OUT} ({len(L)} lines)")
