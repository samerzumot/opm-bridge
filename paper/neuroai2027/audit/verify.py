"""Verify every quantitative claim in short_paper.tex against saved results.

Recomputes the paired contrasts from per-participant records rather than
trusting the pre-aggregated summary fields, then asserts that each value the
manuscript quotes is present in the source. Does not replay raw data.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import subprocess
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
TEX_SOURCE = (BASE / "short_paper.tex").read_text()
MACROS = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^{}]*)\}", (BASE / "factorial_values.tex").read_text()))
TEX = TEX_SOURCE
for name in sorted(MACROS, key=len, reverse=True):
    TEX = TEX.replace("\\" + name, MACROS[name])
assert not re.search(r"\\(?:Fx|Sa)[A-Za-z]+", TEX), "unresolved factorial macro"
SUPP_TABLES = (BASE / "supplement_tables.tex").read_text()


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

# Geometry consistency counts follow the stated negative direction everywhere.
for mode, macro, table_label, supplement_label in (
    ("zero_orientation", "SaZeroOrientationCount", "Zero orientation", "Zero orientation"),
    ("flip_orientation", "SaFlipOrientationCount", "Reversed orient.", "Reversed orientation"),
    ("scramble_channels", "SaScrambleChannelsCount", "Scrambled geometry", "Scrambled geometry"),
):
    delta = sa[mode] - sa["default"]
    count = f"{int((delta < 0).sum())}/{len(delta)}"
    assert MACROS[macro] == count
    row = next(line for line in TEX.splitlines() if line.startswith(table_label + " &"))
    assert f"({count})" in row, row
    row = next(line for line in SUPP_TABLES.splitlines() if line.startswith(supplement_label + " &"))
    assert f" & {count} & " in row, row
    assert TEX_SOURCE.count("\\" + macro) == 2, "count must occur in prose and table"

# --- triaxial cohort --------------------------------------------------------
tri = json.loads((ROOT / "results/summary_phase4_nottingham_full.json").read_text())
assert tri["n_subjects"] == 10 and tri["trials_per_subject"] == 84
subjects = sorted(tri["per_subject"])
pcc = {
    mode: np.asarray([tri["per_subject"][s][mode]["zero_shot_tokenizer_pcc"] for s in subjects])
    for mode in tri["cohort_bootstrap_summary"]
}
# Historical source integrity remains checked below, but the submitted figures
# and tables use only the current full-array factorial.
SIDECAR = json.loads((BASE / "figures/transfer_overview_values.json").read_text())
sidecar_dec = {row["row"]: row for row in SIDECAR["decoding"]}

# --- SQUID control ----------------------------------------------------------
squid = json.loads((ROOT / "results/summary_ds000117_squid_baseline.json").read_text())["reconstruction_pcc"]
check(squid["mag_102ch"]), check(squid["mag_as_grad"])
assert "73.7" in TEX

# --- submission hygiene -----------------------------------------------------
assert chr(0x2014) not in TEX, "em dash present"
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
# the supplement states the architecture as the vendored code has it
assert "causal=False" in SUPP and "masked-token" in SUPP
assert "causal Transformer" not in SUPP
assert "not across trials" in SUPP, "within-trial attention scope must be stated"

# the supplement carries no fabricated cohort it did not run
assert "Emotion" not in SUPP and "emotion" not in SUPP

# anonymity
for token in ("srzumot", "samerzumot", "Marzieh", "Samer", "github.com/samerzumot", "@gmail"):
    assert token not in SUPP and token not in SUPP_TABLES, token
assert chr(0x2014) not in SUPP + SUPP_TABLES, "em dash in supplement"

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

# The noise-floor claim must not outrun the intervals (recomputed below).
assert "not at a dataset noise floor" not in TEX, "noise-floor claim overruns the evidence"
assert "51.03" in TEX, "the chance-excluding classical interval must be quoted"

# the decomposition is described, not claimed as causal
assert "descriptive" in TEX or "algebraic partition" in TEX

# Internal provenance records must not appear in the submitted supplement.
for token in ("tab:s-hashes", "tab:s-environment", "SHA-256", "author-reported", "Scope of verification"):
    assert token not in SUPP + SUPP_TABLES, f"bookkeeping remains in supplement: {token}"

# --- new factorial: independently recompute from the trial arrays -----------
source = ROOT / "results/summary_phase4_nottingham_factorial_and_seeds.json"
new = json.loads(source.read_text())
report = json.loads((BASE / "audit/factorial-results.json").read_text())
assert report["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
assert subjects == sorted(new["per_subject"]) == report["subjects"]
seeds = ["42"] + [str(i) for i in range(101, 110)]
assert report["seeds"] == seeds
fx, bw, dc, seed_scores = {}, {}, {}, {}
record_diffs = []
for f, tag in itertools.product(("raw", "softhfc"), ("mag", "grad")):
    for sub in subjects:
        block = new["per_subject"][sub][f][tag]
        assert set(block["random_partitions"]) == set(seeds)
        for r in [block[m] for m in ("monolithic", "axis_partitioned", "radial_only_z", "tangential_only_xy")] + list(block["random_partitions"].values()):
            v = np.asarray(r["trial_pccs"])
            assert len(v) == 84 and np.isfinite(v).all()
            assert abs(v.mean() - r["pcc_trial_weighted"]) < 1e-12
            b = np.mean([v[i:i+16].mean() for i in range(0, 84, 16)])
            assert abs(b - r["pcc_batch_weighted"]) < 3e-8
            record_diffs.append(abs(v.mean() - r["pcc_batch_weighted"]))
    for mode in ("monolithic", "random", "axis_partitioned"):
        key = f"{f}_{tag}_{mode}"
        groups = [[new["per_subject"][sub][f][tag]["random_partitions"][seed] for seed in seeds] if mode == "random" else [new["per_subject"][sub][f][tag][mode]] for sub in subjects]
        fx[key] = np.array([np.mean([np.mean(r["trial_pccs"]) for r in group]) for group in groups])
        bw[key] = np.array([np.mean([r["pcc_batch_weighted"] for r in group]) for group in groups])
        dc[key] = [100*np.array([group[0]["braille_5fold_bacc"][arm] for group in groups]) for arm in ("pretrained_linear_probe", "control_a_linear_probe", "classical_bandpower")]
        assert np.allclose(fx[key], report["per_participant"][key], atol=1e-14, rtol=0)
    seed_scores[f"{f}_{tag}"] = np.array([[np.mean(new["per_subject"][sub][f][tag]["random_partitions"][seed]["trial_pccs"]) for sub in subjects] for seed in seeds])

contrasts = {}
for f, tag in itertools.product(("raw", "softhfc"), ("mag", "grad")):
    contrasts[f"{f}_{tag}_grouping"] = fx[f"{f}_{tag}_axis_partitioned"] - fx[f"{f}_{tag}_random"]
    contrasts[f"{f}_{tag}_mono_to_random"] = fx[f"{f}_{tag}_random"] - fx[f"{f}_{tag}_monolithic"]
for f in ("raw", "softhfc"):
    for mode in ("monolithic", "random", "axis_partitioned"):
        contrasts[f"{f}_tag_{mode}"] = fx[f"{f}_grad_{mode}"] - fx[f"{f}_mag_{mode}"]
    contrasts[f"{f}_interaction"] = contrasts[f"{f}_grad_grouping"] - contrasts[f"{f}_mag_grouping"]
    assert (contrasts[f"{f}_interaction"] < 0).all()

for family, values in (("cohort", fx), ("contrasts", contrasts)):
    for key, v in values.items():
        r = report[family][key]
        lo, hi = bootstrap_ci(v)
        expected = dict(mean=float(v.mean()), ci_low=lo, ci_high=hi, positive=int((v > 0).sum()), n=10, p=exact_sign_flip_p(v))
        assert all(abs(r[k] - val) < 1e-12 for k, val in expected.items()), key
        name = "Fx" + "".join(part.title() for part in key.split("_"))
        fmt = "+.4f" if family == "contrasts" else ".4f"
        for suffix, text in (("Mean", format(v.mean(), fmt)), ("CI", f"[{lo:{fmt}}, {hi:{fmt}}]"), ("Count", f"{int((v>0).sum())}/10"), ("P", f"{expected['p']:.3f}")):
            assert MACROS[name + suffix] == text, (name, suffix)
        # Check every supplement estimate and interval, including sign and precision.
        assert format(v.mean(), fmt) in SUPP_TABLES and f"[{lo:{fmt}}, {hi:{fmt}}]" in SUPP_TABLES, key

for f, tag in itertools.product(("raw", "softhfc"), ("mag", "grad")):
    key = f"{f}_{tag}"
    per_seed = fx[f"{key}_axis_partitioned"] - seed_scores[key]
    assert (per_seed > 0).all(), key
    means = per_seed.mean(axis=1)
    recorded = report["seed_sensitivity"][key]
    assert np.allclose(recorded["means"], means, atol=1e-14, rtol=0)
    for field, value in (("minimum", means.min()), ("maximum", means.max()), ("sd", means.std(ddof=0))):
        assert abs(recorded[field] - value) < 1e-12
        assert MACROS["Fx" + f.title() + tag.title() + field.title()] == f"{value:.4f}"
    cap = contrasts[f"{key}_mono_to_random"].mean()
    group = contrasts[f"{key}_grouping"].mean()
    for suffix, val in (("MonoToRandomPercent", 100*cap/(cap+group)), ("GroupingPercent", 100*group/(cap+group))):
        assert MACROS["Fx" + f.title() + tag.title() + suffix] == f"{val:.1f}"

max_cohort_weight = max(abs(fx[k].mean()-bw[k].mean()) for k in fx)
assert abs(report["weighting_max_abs_cohort"] - max_cohort_weight) < 1e-12
assert abs(report["weighting_max_abs_record_including_subsets"] - max(record_diffs)) < 1e-12
assert float(MACROS["FxWeightingBound"]) == np.ceil(max_cohort_weight*10000)/10000
for stale in ("one random-partition seed", "Trial-weighted aggregation is a reproducibility check we have not run", "load-bearing free parameter"):
    assert stale not in TEX_SOURCE + SUPP
assert "configuration-dependent" in TEX and "equivalence" in TEX and "unadjusted" in TEX

# Four common MAG full-array conditions replicate at batch weighting.
for old, key in (("triaxial_xyz_raw", "raw_mag_monolithic"), ("triaxial_xyz_softhfc", "softhfc_mag_monolithic"), ("axis_split_raw", "raw_mag_axis_partitioned"), ("phase5_axis_split_xyz", "softhfc_mag_axis_partitioned")):
    assert np.max(abs(pcc[old] - bw[key])) < 5e-8

# Subsets differ in preprocessing, not sample membership; new subset classical
# features are full-array features and must never enter a subset decoding table.
for mode in ("radial_only_z", "tangential_only_xy"):
    for sub in subjects:
        assert tri["per_subject"][sub][mode]["n_channels"] == new["per_subject"][sub]["raw"]["mag"][mode]["n_channels"]
    for f, tag in itertools.product(("raw", "softhfc"), ("mag", "grad")):
        assert all(new["per_subject"][sub][f][tag][mode]["braille_5fold_bacc"]["classical_bandpower"] == new["per_subject"][sub][f][tag]["monolithic"]["braille_5fold_bacc"]["classical_bandpower"] for sub in subjects)

# Independently verify the new decoding table rows and inferential scope.
for key, (probe, ctrl, classical) in dc.items():
    pl, ph = bootstrap_ci(probe); cl, ch = bootstrap_ci(classical)
    assert pl <= 50 <= ph and classical.mean() > probe.mean(), key
    row = f"{probe.mean():.2f} [{pl:.2f}, {ph:.2f}] & {ctrl.mean():.2f} & {classical.mean():.2f} [{cl:.2f}, {ch:.2f}] & {int((classical > probe).sum())}/10"
    assert row in SUPP_TABLES, key
assert "full-array factorial" in TEX
assert "Ctrl A denotes the random-initialization control" in SUPP_TABLES

# All numerical panels, including the new four grouping contrasts.
for row, key in zip(SIDECAR["triaxial_contrasts"], ("softhfc_mag_grouping", "raw_mag_grouping", "softhfc_grad_grouping", "raw_grad_grouping")):
    expected = report["contrasts"][key]
    for name in ("mean", "ci_low", "ci_high", "n"):
        assert abs(row[name]-expected[name]) < 1e-12
    assert row["consistent"] == expected["positive"]
for row, mode in zip(SIDECAR["single_axis_contrasts"], ("opm_as_grad", "zero_orientation", "flip_orientation", "scramble_channels")):
    d = sa[mode]-sa["default"]
    lo, hi = bootstrap_ci(d)
    assert np.allclose([row["mean"], row["ci_low"], row["ci_high"]], [d.mean(), lo, hi], rtol=0, atol=1e-12)
    assert row["consistent"] == int((d * np.sign(d.mean()) > 0).sum())
    assert row["direction"] == ("positive" if d.mean() > 0 else "negative")
for mode in ("mag_102ch", "grad_204ch", "mag_as_grad"):
    assert SIDECAR["squid_control"][mode] == squid[mode]
for key, arr in (("single_axis_default", sa["default"]), ("single_axis_opm_as_grad", sa["opm_as_grad"]), ("triaxial_monolithic_softhfc", fx["softhfc_mag_monolithic"]), ("triaxial_random_softhfc", fx["softhfc_mag_random"]), ("triaxial_axis_split", fx["softhfc_mag_axis_partitioned"])):
    assert abs(SIDECAR["slopegraph_means"][key] - arr.mean()) < 1e-12

for path in (BASE / "short_paper.tex", BASE / "supplement.tex", BASE / "supplement_tables.tex", BASE / "factorial_values.tex"):
    text = path.read_text()
    assert chr(0x2014) not in text
    assert all(token not in text for token in ("Samer", "Marzieh", "samerzumot", "@gmail"))
title = "Sensor Metadata and Axis-Aware Tokenization Shape Zero-Shot Reconstruction in OPM-MEG"
assert title in TEX_SOURCE and title in SUPP
for forbidden in ("dissociat", "Capacity estimates", "capacity fraction", "Capacity: random", "showing that reconstruction gains do not propagate"):
    assert forbidden not in TEX + SUPP + SUPP_TABLES, forbidden
assert "We observe improved reconstruction without a detectable improvement in linear decoding." in TEX
assert "attention composition as well as query budget" in TEX and "attention composition as well as query budget" in SUPP

# Row-level supplement verification: values cannot silently migrate to another label.
labels = {"monolithic": "Monolithic", "random": "Random", "axis_partitioned": "Axis"}
for f in ("raw", "softhfc"):
    filter_label = "Raw" if f == "raw" else "Soft-HFC"
    for tag in ("mag", "grad"):
        for mode, label in labels.items():
            v = fx[f"{f}_{tag}_{mode}"]; lo, hi = bootstrap_ci(v)
            row = f"{filter_label} & {tag.upper()} & {label} & {v.mean():.4f} & [{lo:.4f}, {hi:.4f}]"
            assert row in SUPP_TABLES, row
            probe, ctrl, classical = dc[f"{f}_{tag}_{mode}"]
            pl, ph = bootstrap_ci(probe); cl, ch = bootstrap_ci(classical)
            row = f"{filter_label} & {tag.upper()} & {label} & {probe.mean():.2f} [{pl:.2f}, {ph:.2f}] & {ctrl.mean():.2f} & {classical.mean():.2f} [{cl:.2f}, {ch:.2f}] & {int((classical>probe).sum())}/10"
            assert row in SUPP_TABLES, row
    rows = [(f"{f}_{t}_{c}", f"{t.upper()}: {lab}") for t in ("mag", "grad") for c, lab in (("grouping", "Axis minus random"), ("mono_to_random", "Random minus mono"))]
    rows += [(f"{f}_tag_{mode}", f"GRAD minus MAG: {label}") for mode, label in labels.items()]
    rows += [(f"{f}_interaction", "Grouping interaction")]
    for key, label in rows:
        v = contrasts[key]; lo, hi = bootstrap_ci(v)
        row = f"{filter_label} & {label} & ${v.mean():+.4f}$ & $[{lo:+.4f}, {hi:+.4f}]$ & {int((v>0).sum())}/10 & {exact_sign_flip_p(v):.2e}"
        assert row in SUPP_TABLES, row
    for i in range(len(subjects)):
        row = f"{filter_label} & P{i+1:02d} & " + " & ".join(f"{fx[f'{f}_{tag}_{mode}'][i]:.4f}" for tag in ("mag", "grad") for mode in labels)
        assert row in SUPP_TABLES, row
for i, seed in enumerate(seeds):
    row = seed + " & " + " & ".join(f"{(fx[f'{f}_{tag}_axis_partitioned']-seed_scores[f'{f}_{tag}'][i]).mean():+.4f}" for f in ("raw", "softhfc") for tag in ("mag", "grad"))
    assert row in SUPP_TABLES, row
row = "SD & " + " & ".join(f"{(fx[f'{f}_{tag}_axis_partitioned']-seed_scores[f'{f}_{tag}']).mean(axis=1).std(ddof=0):.4f}" for f in ("raw", "softhfc") for tag in ("mag", "grad"))
assert row in SUPP_TABLES
# Figure F contains the six current Soft-HFC cells, with every marker, interval
# and paired consistency count independently checked against the source.
expected_figure_rows = set()
for tag, mode in itertools.product(("mag", "grad"), ("monolithic", "random", "axis_partitioned")):
    label = tag.upper() + " " + {"monolithic": "mono", "random": "rand", "axis_partitioned": "axis"}[mode]
    expected_figure_rows.add(label)
    row = sidecar_dec[label]
    probe, _, classical = dc[f"softhfc_{tag}_{mode}"]
    assert abs(row["probe"]-probe.mean()) < 1e-12
    assert abs(row["classical"]-classical.mean()) < 1e-12
    assert np.allclose(row["probe_ci"], bootstrap_ci(probe), atol=1e-12, rtol=0)
    assert np.allclose(row["classical_ci"], bootstrap_ci(classical), atol=1e-12, rtol=0)
    assert row["consistent"] == int((classical > probe).sum()) and row["n"] == 10
assert set(sidecar_dec) == expected_figure_rows and len(SIDECAR["decoding"]) == 6

# Current decoding summaries, classical invariance, and all paired grouping
# comparisons support the narrower decoding claim without interpreting a null
# as equivalence or an endpoint interaction.
for key, arrays in dc.items():
    for arm, v in zip(("pretrained_linear_probe", "control_a_linear_probe", "classical_bandpower"), arrays):
        lo, hi = bootstrap_ci(v)
        expected = dict(mean=float(v.mean()), ci_low=lo, ci_high=hi, positive=int((v>0).sum()), n=10, p=exact_sign_flip_p(v))
        assert all(abs(report["decoding"][key][arm][k]-value)<1e-12 for k,value in expected.items()), (key,arm)
probe_means = [v[0].mean() for v in dc.values()]
assert MACROS["FxProbeMinimum"] == f"{min(probe_means):.2f}"
assert MACROS["FxProbeMaximum"] == f"{max(probe_means):.2f}"
for f in ("raw", "softhfc"):
    classical = dc[f"{f}_mag_monolithic"][2]
    lo, hi = bootstrap_ci(classical)
    assert MACROS["FxClassical"+f.title()+"Mean"] == f"{classical.mean():.2f}"
    assert MACROS["FxClassical"+f.title()+"CI"] == f"[{lo:.2f}, {hi:.2f}]"
    assert (lo>50) == (f=="softhfc")
    for tag in ("mag", "grad"):
        for mode in labels:
            assert np.array_equal(dc[f"{f}_{tag}_{mode}"][2], classical), (f,tag,mode)
        for mode, label in (("monolithic", "Axis minus mono"), ("random", "Axis minus random")):
            v = dc[f"{f}_{tag}_axis_partitioned"][0]-dc[f"{f}_{tag}_{mode}"][0]
            lo, hi = bootstrap_ci(v)
            p_value = exact_sign_flip_p(v)
            assert lo<=0<=hi and p_value>0.05, (f,tag,mode)
            row = f"{'Raw' if f=='raw' else 'Soft-HFC'} & {tag.upper()} & {label} & ${v.mean():+.4f}$ & $[{lo:+.4f}, {hi:+.4f}]$ & {int((v>0).sum())}/10 & {p_value:.2e}"
            assert row in SUPP_TABLES, row

# Submission contains final scientific methods/results, not superseded grids,
# invalid tasks, subset comparisons, or the analysis history. Provenance stays
# in audit/ and the immutable source records.
for token in ("withdraw", "evoked-detection", "Original Triaxial", "original MAG", "New full-array", "subset-scope", "Why the subset", "95.39", "99.81", "tab:s-decoding}", "tab:s-tri-cohort", "tab:s-tri-contrasts", "tab:s-tri-per-subject"):
    assert token not in TEX + SUPP + SUPP_TABLES, token
expected_tables = ["single-per-subject", "single-contrasts", "factorial-cohort", "factorial-contrasts", "factorial-participants", "factorial-seeds", "factorial-decoding", "decoding-contrasts"]
assert re.findall(r"\\label\{tab:s-([^}]+)\}", SUPP_TABLES) == expected_tables
assert "Tables S3--S6" in TEX_SOURCE and "Table S7" in TEX_SOURCE and "Table S8" in TEX_SOURCE

# Bibliography integrity and attribution checks complement the documented
# human full-text review; these checks alone do not establish source support.
bib = (BASE / "references.bib").read_text()
bib_keys = re.findall(r"@[A-Za-z]+\s*\{\s*([^,]+),", bib)
assert len(bib_keys) == len(set(bib_keys)), "duplicate bibliography keys"
new_keys = {"jiang2024labram", "rea2022triaxial", "tierney2021hfc"}
assert new_keys <= set(bib_keys)
citation_review = json.loads((BASE / "audit/reference-audit.json").read_text())
reviewed = {entry["key"]: entry for entry in citation_review["entries"]}
for key in new_keys:
    assert reviewed[key]["status"] == "supported"
    assert reviewed[key]["source_locators"] and reviewed[key]["sources"]
# The 2026-11-11 Soft-HFC practice citations were verified at documentation and metadata
# level, not against full text, and are cited only for the existence of a reference
# implementation and of the harmonic generalization. Pin that so the weaker basis is not
# later reported as full-text support.
practice_keys = {"mneopmtutorial", "seymour2022interference", "tierney2022harmonic"}
assert practice_keys <= set(bib_keys)
for key in practice_keys:
    assert reviewed[key]["status"] in {"metadata_and_documentation_verified",
                                       "metadata_verified_not_full_text"}
    assert reviewed[key]["source_locators"] and reviewed[key]["sources"] and reviewed[key]["limits"]
for name, document in (("short_paper", TEX_SOURCE), ("supplement", SUPP)):
    citations = {key.strip() for group in re.findall(r"\\cite[pt]?\*?(?:\[[^\]]*\])*\{([^}]+)\}", document) for key in group.split(",")}
    assert citations <= set(bib_keys), (name, citations-set(bib_keys))
    bbl = (BASE / f"build/{name}.bbl").read_text()
    printed = set(re.findall(r"\\bibitem(?:\[[\s\S]*?\])?\{([^{}]+)\}", bbl))
    assert printed == citations, (name, printed, citations)
    blg = (BASE / f"build/{name}.blg").read_text()
    assert "Warning--" not in blg and "error message" not in blg
    assert chr(0x2014) not in bbl, name
    if name == "short_paper":
        assert citations == set(bib_keys), "uncited entries in the submission bibliography"
        assert new_keys <= citations
    else:
        assert {"tierney2021hfc", "xiao2025brainomni"} <= citations
assert r"a soft variant of homogeneous field correction \citep{tierney2021hfc}" in TEX_SOURCE
assert r"not a value established by \citet{tierney2021hfc}" in SUPP
assert "we approximate distant magnetic interference" in SUPP
assert "full column rank" in SUPP and "In the evaluated configuration" in SUPP
for unsupported in ("trained almost entirely", "more than four to one", r"20--40\,mm", r"roughly 5\,mm", "at one coordinate"):
    assert unsupported not in TEX_SOURCE, unsupported
print(f"PASS: {len(bib_keys)} unique references; both compiled bibliographies match their citations, and every added citation has an audit record at the level its entry states.")

# Page budget and visible PDF text, in addition to TeX source checks.
paper_pages = subprocess.check_output(["pdftotext", str(BASE / "build/short_paper.pdf"), "-"], text=True).split("\f")
paper_pages = [p for p in paper_pages if p.strip()]
reference_starts = [(i, re.search(r"(?m)^References\s*$", page)) for i, page in enumerate(paper_pages)]
reference_starts = [(i, match) for i, match in reference_starts if match]
assert len(reference_starts) == 1, "expected one references section"
ref_page, heading = reference_starts[0]
assert ref_page <= 4, "content exceeds four pages"
if ref_page == 4:
    assert not paper_pages[ref_page][:heading.start()].strip(), "content spills onto page five"
# References can span additional pages; the workshop excludes them from the limit.
assert re.search(r"\\bibliography\{references\}\s*\\end\{document\}", TEX_SOURCE)

for filename in ("short_paper", "supplement"):
    rendered = subprocess.check_output(["pdftotext", str(BASE / f"build/{filename}.pdf"), "-"], text=True)
    assert "??" not in rendered, f"unresolved reference in {filename}"
    assert all(token not in rendered for token in ("Samer", "Marzieh", "samerzumot", "@gmail"))
print(f"PASS: paper has {len(paper_pages)} total pages; references begin on page {ref_page+1} and content fits four pages. PDF text is anonymous and references resolve.")
print("PASS: final scientific narrative uses the current full-array factorial; 560 trial records, all factorial estimates, interactions, seed robustness, weighting scope, full-array decoding, figure sidecar, generated macros, supplement, anonymity and build logs verified.")
