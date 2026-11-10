"""Generate scientific supplement tables from saved participant results."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from factorial_results import write as write_factorial, SEEDS

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

# Full-array factorial.
report, fx, delta = write_factorial()
data = json.loads((RESULTS / "summary_phase4_nottingham_factorial_and_seeds.json").read_text())
subjects = sorted(data["per_subject"])
labels = {"monolithic": "Monolithic", "random": "Random", "axis_partitioned": "Axis"}

def start_table(columns, header):
    A(r"\begin{table*}[htbp]\centering\small")
    A(r"\begin{tabular}{" + columns + "}")
    A(r"\toprule")
    A(header + r" \\")
    A(r"\midrule")

def finish_table(caption, label):
    A(r"\bottomrule\end{tabular}")
    A(r"\caption{" + caption + "}")
    A(r"\label{" + label + r"}\end{table*}")
    A("")

start_table("lllrr", r"Filter & Tag & Partition & Mean PCC & 95\% CI")
for f in ("raw", "softhfc"):
    for t in ("mag", "grad"):
        for m, label in labels.items():
            v = report["cohort"][f"{f}_{t}_{m}"]
            A(f"{'Raw' if f == 'raw' else 'Soft-HFC'} & {t.upper()} & {label} & {v['mean']:.4f} & [{v['ci_low']:.4f}, {v['ci_high']:.4f}]" + r" \\")
finish_table(r"Full-array factorial reconstruction. Trial-weighted participant PCC; random scores average ten seeds within participant. All intervals use 20{,}000 participant bootstrap draws with seed 0. Neither seeds nor trials are treated as independent participants.", "tab:s-factorial-cohort")

start_table("llrrrr", r"Filter & Contrast & Mean $\Delta$ & 95\% CI & Positive & Exact $p$")
for f in ("raw", "softhfc"):
    rows = [(f"{f}_{t}_{c}", f"{t.upper()}: {lab}") for t in ("mag", "grad") for c, lab in (("grouping", "Axis minus random"), ("mono_to_random", "Random minus mono"))]
    rows += [(f"{f}_tag_{m}", f"GRAD minus MAG: {label}") for m, label in labels.items()]
    rows += [(f"{f}_interaction", "Grouping interaction")]
    for key, label in rows:
        A(("Raw" if f == "raw" else "Soft-HFC") + " & " + contrast_row(label, delta[key], True))
finish_table(r"Paired factorial contrasts. Positive counts strictly positive participant differences, including when the mean is negative. The grouping interaction is $(A-R)_{\mathrm{GRAD}}-(A-R)_{\mathrm{MAG}}$, with $R$ averaged across seeds within participant. Exact two-sided sign-flip tests are exploratory and unadjusted for multiplicity.", "tab:s-factorial-contrasts")

start_table("llrrrrrr", r"Filter & Participant & MAG mono & MAG rand & MAG axis & GRAD mono & GRAD rand & GRAD axis")
for f in ("raw", "softhfc"):
    for i, sub in enumerate(subjects):
        A(("Raw" if f == "raw" else "Soft-HFC") + f" & P{i+1:02d} & " + " & ".join(f"{fx[f'{f}_{t}_{m}'][i]:.4f}" for t in ("mag", "grad") for m in labels) + r" \\")
finish_table(r"Full-array factorial PCC per participant. Indices follow sorted dataset participant IDs. Random columns average the ten saved seeds; other columns use one condition per participant.", "tab:s-factorial-participants")

start_table("lrrrr", r"Seed & Raw MAG & Raw GRAD & Soft-HFC MAG & Soft-HFC GRAD")
for i, seed in enumerate(SEEDS):
    A(seed + " & " + " & ".join(f"{report['seed_sensitivity'][f'{f}_{t}']['means'][i]:+.4f}" for f in ("raw", "softhfc") for t in ("mag", "grad")) + r" \\")
A(r"\midrule")
A("SD & " + " & ".join(f"{report['seed_sensitivity'][f'{f}_{t}']['sd']:.4f}" for f in ("raw", "softhfc") for t in ("mag", "grad")) + r" \\")
finish_table(r"Seed-specific cohort axis-minus-random PCC contrasts, trial-weighted. SD is the descriptive population SD across the ten observed seed-specific means (divisor ten), not a standard error. Group sizes exactly match axis counts in the generating script. Every participant contrast is positive for every seed in all four tag/filter combinations.", "tab:s-factorial-seeds")

start_table("lllrrrr", r"Filter & Tag & Partition & Probe [95\% CI] & Ctrl A & Classical [95\% CI] & Cl. $>$ Pr.")
for f in ("raw", "softhfc"):
    for t in ("mag", "grad"):
        for m, label in labels.items():
            vals = []
            for arm, _ in DEC_ARMS:
                vals.append(100*np.asarray([data['per_subject'][sub][f][t]['random_partitions']['42']['braille_5fold_bacc'][arm] if m == 'random' else data['per_subject'][sub][f][t][m]['braille_5fold_bacc'][arm] for sub in subjects]))
            pr, ctrl, cl = vals
            pl, ph = bootstrap_ci(pr); cll, clh = bootstrap_ci(cl)
            assert pl <= 50 <= ph and cl.mean() > pr.mean()
            A(f"{'Raw' if f == 'raw' else 'Soft-HFC'} & {t.upper()} & {label} & {pr.mean():.2f} [{pl:.2f}, {ph:.2f}] & {ctrl.mean():.2f} & {cl.mean():.2f} [{cll:.2f}, {clh:.2f}] & {int((cl > pr).sum())}/{len(subjects)}" + r" \\")
finish_table(r"Full-array factorial Braille balanced accuracy (\%), with participant bootstrap intervals. Ctrl A denotes the random-initialization control. Random decoding is available only at seed 42, with group sizes exactly matched to the axes; reconstruction averages ten seeds. Every pretrained-probe interval includes chance; this does not establish equivalence.", "tab:s-factorial-decoding")

start_table("lllrrrr", r"Filter & Tag & Probe contrast & Mean $\Delta$ & 95\% CI & Positive & Exact $p$")
for f in ("raw", "softhfc"):
    for t in ("mag", "grad"):
        scores = {}
        for m in labels:
            scores[m] = 100*np.asarray([(data['per_subject'][sub][f][t]['random_partitions']['42'] if m == 'random' else data['per_subject'][sub][f][t][m])['braille_5fold_bacc']['pretrained_linear_probe'] for sub in subjects])
        for m, label in (("monolithic", "Axis minus mono"), ("random", "Axis minus random")):
            A(("Raw" if f == "raw" else "Soft-HFC") + " & " + t.upper() + " & " + contrast_row(label, scores['axis_partitioned']-scores[m], True))
finish_table(r"Paired pretrained-probe decoding contrasts in balanced-accuracy percentage points, ten participants. Random partitions use seed 42. Intervals use 20{,}000 participant bootstrap draws; exact two-sided sign-flip tests are exploratory and unadjusted for multiplicity. Positive counts strictly positive participant differences. Every interval includes zero; this does not establish equivalence.", "tab:s-decoding-contrasts")

OUT.write_text("\n".join(L) + "\n")
print(f"wrote {OUT} ({len(L)} lines)")
