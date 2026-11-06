"""Verify every quantitative claim in short_paper.tex against saved results.

Recomputes the paired contrasts from per-participant records rather than
trusting the pre-aggregated summary fields, then asserts that each value the
manuscript quotes is present in the source. Does not replay raw data.
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
TEX = (BASE / "short_paper.tex").read_text()
LOG = (BASE / "build/short_paper.log").read_text()


def present(value: str) -> bool:
    """A number is quoted either in full or with the leading zero dropped."""
    return value in TEX or value.lstrip("0") in TEX


def check(value: float, digits: int = 4) -> None:
    text = f"{value:.{digits}f}"
    assert present(text), f"value absent from manuscript: {text}"


def bootstrap_ci(values: np.ndarray, n_boot: int = 20000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = values[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def exact_sign_flip_p(delta: np.ndarray) -> float:
    n = len(delta)
    observed = abs(delta.mean())
    hits = sum(
        1
        for signs in itertools.product((1, -1), repeat=n)
        if abs((delta * np.array(signs)).mean()) >= observed - 1e-15
    )
    return hits / 2**n


# --- single-axis cohort -----------------------------------------------------
single = json.loads((ROOT / "results/summary_phase3_ablations.json").read_text())
single = single["zero_shot_tokenizer_pcc_by_geometry_mode"]
sa = {k: np.asarray(v["per_subject_pcc"], dtype=float) for k, v in single.items()}
assert all(len(v) == 21 for v in sa.values()), "expected 21 single-axis participants"

for mode in ("default", "opm_as_grad", "zero_orientation", "flip_orientation", "scramble_channels"):
    check(single[mode]["mean_pcc"])
    check(single[mode]["ci_low"])
    check(single[mode]["ci_high"])

remap = sa["opm_as_grad"] - sa["default"]
assert (remap > 0).all(), "remapping claimed consistent in 21/21 participants"
check(remap.mean())
lo, hi = bootstrap_ci(remap)
check(lo), check(hi)
assert exact_sign_flip_p(remap) < 1e-5
assert abs(100 * remap.mean() / sa["default"].mean() - 52.7) < 0.1

for mode, expect_pos in (("zero_orientation", 0), ("flip_orientation", 0), ("scramble_channels", 2)):
    delta = sa[mode] - sa["default"]
    assert int((delta > 0).sum()) == expect_pos, (mode, int((delta > 0).sum()))
    check(delta.mean())

# --- triaxial cohort --------------------------------------------------------
tri = json.loads((ROOT / "results/summary_phase4_nottingham_full.json").read_text())
assert tri["n_subjects"] == 10 and tri["trials_per_subject"] == 84
subjects = sorted(tri["per_subject"])
pcc = {
    mode: np.asarray([tri["per_subject"][s][mode]["zero_shot_tokenizer_pcc"] for s in subjects])
    for mode in tri["cohort_bootstrap_summary"]
}
# one source of truth: the manuscript table quotes intervals recomputed here from
# the per-participant records, not the intervals saved by the benchmark run
for mode in tri["cohort_bootstrap_summary"]:
    check(pcc[mode].mean())
    lo, hi = bootstrap_ci(pcc[mode])
    check(lo), check(hi)

# every contrast the manuscript quotes, each within one filtering regime
CONTRASTS = {
    "axis_vs_random_softhfc": (pcc["phase5_axis_split_xyz"] - pcc["random_3way_split_softhfc"], 1),
    "axis_vs_random_raw": (pcc["axis_split_raw"] - pcc["random_3way_split"], 1),
    "random_vs_mono_softhfc": (pcc["random_3way_split_softhfc"] - pcc["triaxial_xyz_softhfc"], 1),
    "random_vs_mono_raw": (pcc["random_3way_split"] - pcc["triaxial_xyz_raw"], 1),
    "softhfc_on_monolithic": (pcc["triaxial_xyz_softhfc"] - pcc["triaxial_xyz_raw"], -1),
    "softhfc_on_random": (pcc["random_3way_split_softhfc"] - pcc["random_3way_split"], -1),
    "softhfc_on_axis": (pcc["phase5_axis_split_xyz"] - pcc["axis_split_raw"], -1),
}
for name, (delta, sign) in CONTRASTS.items():
    assert ((delta * sign) > 0).all(), f"{name} claimed consistent in 10/10 participants"
    assert abs(exact_sign_flip_p(delta) - 2 / 2**10) < 1e-9, name
    check(delta.mean())

# the four grouping and capacity contrasts also quote their intervals
for name in ("axis_vs_random_softhfc", "axis_vs_random_raw",
             "random_vs_mono_softhfc", "random_vs_mono_raw"):
    lo, hi = bootstrap_ci(CONTRASTS[name][0])
    check(lo), check(hi)

# the decomposition percentages are an algebraic partition of the total gain
for regime, cap_name, grp_name, total in (
    ("softhfc", "random_vs_mono_softhfc", "axis_vs_random_softhfc",
     pcc["phase5_axis_split_xyz"] - pcc["triaxial_xyz_softhfc"]),
    ("raw", "random_vs_mono_raw", "axis_vs_random_raw",
     pcc["axis_split_raw"] - pcc["triaxial_xyz_raw"]),
):
    cap, grp = CONTRASTS[cap_name][0].mean(), CONTRASTS[grp_name][0].mean()
    assert abs(cap + grp - total.mean()) < 1e-12, regime
    assert f"{100 * cap / total.mean():.1f}" in TEX, regime
    assert f"{100 * grp / total.mean():.1f}" in TEX, regime

# the additivity estimate is withdrawn: the grid measures it directly
assert "additivity assumption" not in TEX or "no longer" in TEX or "rather than on an additivity" in TEX or "rests on a complete" in TEX

# --- decoding table ---------------------------------------------------------
def bacc(mode: str, arm: str) -> float:
    return float(
        np.mean([tri["per_subject"][s][mode]["braille_5fold_bacc"][arm] for s in subjects])
    )


SIDECAR = json.loads((BASE / "figures/transfer_overview_values.json").read_text())
sidecar_dec = {row["row"]: row for row in SIDECAR["decoding"]}
DEC_ROWS = {
    "triaxial_xyz_raw": "Monolithic, raw",
    "triaxial_xyz_softhfc": "Monolithic, HFC",
    "random_3way_split": "Random, raw",
    "random_3way_split_softhfc": "Random, HFC",
    "axis_split_raw": "Axis, raw",
    "phase5_axis_split_xyz": "Axis, HFC",
    "radial_only_z": "Radial, raw",
    "tangential_only_xy": "Tangential, raw",
}
assert set(DEC_ROWS) == set(tri["cohort_bootstrap_summary"]), "decoding rows must cover every mode"
for mode, row_name in DEC_ROWS.items():
    probe, classical = bacc(mode, "pretrained_linear_probe"), bacc(mode, "classical_bandpower")
    assert classical > probe, f"classical baseline claimed to lead in {mode}"
    # the decoding values are carried by Figure 1F, so check the figure sidecar
    row = sidecar_dec[row_name]
    assert abs(row["probe"] - probe * 100) < 1e-6, row_name
    assert abs(row["classical"] - classical * 100) < 1e-6, row_name
    per_sub_probe = np.asarray(
        [tri["per_subject"][s][mode]["braille_5fold_bacc"]["pretrained_linear_probe"] for s in subjects]
    )
    per_sub_class = np.asarray(
        [tri["per_subject"][s][mode]["braille_5fold_bacc"]["classical_bandpower"] for s in subjects]
    )
    assert row["consistent"] == int((per_sub_class > per_sub_probe).sum()), row_name
    assert row["consistent"] in (7, 8), (row_name, row["consistent"])

# the range endpoints and the invariance claim quoted in the text
for quoted in ("47.19", "50.83", "52.76", "54.50", "53.54", "53.33", "7 or 8 of 10"):
    assert quoted in TEX, quoted

# the classical arm depends only on channel set and filtering, not on partition
for a, b in (("random_3way_split", "triaxial_xyz_raw"),
             ("axis_split_raw", "triaxial_xyz_raw"),
             ("random_3way_split_softhfc", "triaxial_xyz_softhfc"),
             ("phase5_axis_split_xyz", "triaxial_xyz_softhfc")):
    assert bacc(a, "classical_bandpower") == bacc(b, "classical_bandpower"), (a, b)
# the probe arm, by contrast, now differs across partitions: the earlier
# feature-reuse defect is fixed upstream
assert bacc("random_3way_split", "pretrained_linear_probe") != bacc(
    "triaxial_xyz_raw", "pretrained_linear_probe"
), "random-partition probe still duplicates the monolithic row"

# --- SQUID control ----------------------------------------------------------
squid = json.loads((ROOT / "results/summary_ds000117_squid_baseline.json").read_text())["reconstruction_pcc"]
check(squid["mag_102ch"]), check(squid["mag_as_grad"])
assert "73.7" in TEX

# --- submission hygiene -----------------------------------------------------
assert "—" not in TEX, "em dash present"
assert all(token not in TEX for token in ("srzumot", "samerzumot", "Marzieh", "Samer", "github.com/samerzumot"))
assert "76.1" not in TEX, "withdrawn causal attribution reintroduced"
assert "Overfull" not in LOG and "undefined" not in LOG

# --- supplement -------------------------------------------------------------
SUPP = (BASE / "supplement.tex").read_text()
SUPP_TABLES = (BASE / "supplement_tables.tex").read_text()

# the generated tables must carry every per-participant value, not a transcription
for mode in ("default", "opm_as_grad", "zero_orientation", "flip_orientation", "scramble_channels"):
    for value in sa[mode]:
        assert f"{value:.4f}" in SUPP_TABLES, f"single-axis value missing: {mode} {value:.4f}"
for mode in tri["cohort_bootstrap_summary"]:
    col = np.asarray([tri["per_subject"][s][mode]["zero_shot_tokenizer_pcc"] for s in subjects])
    for value in col:
        assert f"{value:.4f}" in SUPP_TABLES, f"triaxial value missing: {mode} {value:.4f}"
for mode in DEC_ROWS:
    for arm in ("pretrained_linear_probe", "control_a_linear_probe", "classical_bandpower"):
        assert f"{bacc(mode, arm) * 100:.2f}" in SUPP_TABLES, (mode, arm)

# the supplement states the architecture as the vendored code has it
assert "causal=False" in SUPP and "masked-token" in SUPP
assert "causal Transformer" not in SUPP
assert "not across trials" in SUPP, "within-trial attention scope must be stated"

# the supplement carries no fabricated cohort it did not run
assert "Emotion" not in SUPP and "emotion" not in SUPP

# anonymity
for token in ("srzumot", "samerzumot", "Marzieh", "Samer", "github.com/samerzumot", "@gmail"):
    assert token not in SUPP and token not in SUPP_TABLES, token
assert "—" not in SUPP, "em dash in supplement"

SUPP_LOG = (BASE / "build/supplement.log").read_text()
assert "Overfull" not in SUPP_LOG and "undefined" not in SUPP_LOG

# --- review findings, as standing checks ------------------------------------
# the classical baseline is six log band powers plus four temporal window means
BASELINE_SRC = (ROOT / "src/opm_bridge/models/baselines.py").read_text()
n_bands = BASELINE_SRC.count("),    #") + BASELINE_SRC.count("),  #")
assert "np.array_split(X, 4, axis=-1)" in BASELINE_SRC, "temporal window count changed"
assert "BANDS = (" in BASELINE_SRC
bands_block = BASELINE_SRC.split("BANDS = (")[1].split(")\n")[0]
assert bands_block.count("(") == 6, f"expected 6 bands, found {bands_block.count('(')}"
for doc in (TEX, SUPP):
    assert "four-band" not in doc, "baseline is six bands plus four window means"
    assert "four spectral features" not in doc
assert "six log band powers" in TEX and "six log band powers" in SUPP

# the noise-floor claim must not outrun the intervals
def bacc_per_subject(mode: str, arm: str) -> np.ndarray:
    return 100.0 * np.asarray(
        [tri["per_subject"][s][mode]["braille_5fold_bacc"][arm] for s in subjects]
    )


classical_excl = [
    m for m in DEC_ROWS
    if bootstrap_ci(bacc_per_subject(m, "classical_bandpower"))[0] > 50.0
]
assert set(classical_excl) == {
    "triaxial_xyz_softhfc", "random_3way_split_softhfc", "phase5_axis_split_xyz"
}, classical_excl
assert "not at a dataset noise floor" not in TEX, "noise-floor claim overruns the evidence"
assert "51.03" in TEX, "the one chance-excluding interval must be quoted"

# the decomposition is described, not claimed as causal
assert "descriptive" in TEX or "algebraic partition" in TEX

# Internal provenance records must not appear in the submitted supplement.
for token in ("tab:s-hashes", "tab:s-environment", "SHA-256", "author-reported", "Scope of verification"):
    assert token not in SUPP + SUPP_TABLES, f"bookkeeping remains in supplement: {token}"

print(
    "PASS: single-axis and triaxial values recomputed from per-participant records; "
    "paired contrasts, consistency counts and permutation p-values match the manuscript; "
    "decoding table, SQUID control, anonymity, em-dash and build checks clean; "
    "supplement tables carry every saved per-participant value and state the "
    "architecture as the vendored code has it."
)
