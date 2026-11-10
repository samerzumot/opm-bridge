"""Generate the factorial report and LaTeX values directly from saved trials.

Random seeds are averaged within participants before inference. Participants,
not seeds or trials, are the independent sampling units. No model inference.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
SOURCE = BASE.parents[1] / "results/summary_phase4_nottingham_factorial_and_seeds.json"
FILTERS = ("raw", "softhfc")
TAGS = ("mag", "grad")
MODES = ("monolithic", "random", "axis_partitioned")
SEEDS = ("42", *map(str, range(101, 110)))


def stats(values):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(0)
    means = v[rng.integers(0, len(v), size=(20000, len(v)))].mean(axis=1)
    signs = np.asarray(list(itertools.product((1, -1), repeat=len(v))))
    return dict(mean=float(v.mean()), ci_low=float(np.percentile(means, 2.5)),
                ci_high=float(np.percentile(means, 97.5)),
                positive=int((v > 0).sum()), n=len(v),
                p=float((abs((signs * v).mean(axis=1)) >= abs(v.mean()) - 1e-15).mean()))


def load():
    data = json.loads(SOURCE.read_text())
    subjects = sorted(data["per_subject"])
    assert len(subjects) == data["n_subjects"] == 10
    arrays, batch, seed_arrays = {}, {}, {}
    all_weighting = []
    for f, t in itertools.product(FILTERS, TAGS):
        for s in subjects:
            block = data["per_subject"][s][f][t]
            assert set(block["random_partitions"]) == set(SEEDS)
            records = [block[m] for m in ("monolithic", "axis_partitioned", "radial_only_z", "tangential_only_xy")]
            records += list(block["random_partitions"].values())
            for rec in records:
                a = np.asarray(rec["trial_pccs"])
                assert len(a) == 84 and np.isfinite(a).all()
                assert abs(a.mean() - rec["pcc_trial_weighted"]) < 1e-12
                b = np.mean([a[i:i+16].mean() for i in range(0, 84, 16)])
                assert abs(b - rec["pcc_batch_weighted"]) < 3e-8
                all_weighting.append(abs(a.mean() - rec["pcc_batch_weighted"]))
        for m in MODES:
            key = f"{f}_{t}_{m}"
            for field, target in (("pcc_trial_weighted", arrays), ("pcc_batch_weighted", batch)):
                target[key] = np.asarray([
                    np.mean([data["per_subject"][s][f][t]["random_partitions"][seed][field] for seed in SEEDS])
                    if m == "random" else data["per_subject"][s][f][t][m][field]
                    for s in subjects])
        seed_arrays[f"{f}_{t}"] = np.asarray([
            [data["per_subject"][s][f][t]["random_partitions"][seed]["pcc_trial_weighted"] for s in subjects]
            for seed in SEEDS])
    contrasts = {}
    for f, t in itertools.product(FILTERS, TAGS):
        contrasts[f"{f}_{t}_grouping"] = arrays[f"{f}_{t}_axis_partitioned"] - arrays[f"{f}_{t}_random"]
        contrasts[f"{f}_{t}_mono_to_random"] = arrays[f"{f}_{t}_random"] - arrays[f"{f}_{t}_monolithic"]
    for f in FILTERS:
        for m in MODES:
            contrasts[f"{f}_tag_{m}"] = arrays[f"{f}_grad_{m}"] - arrays[f"{f}_mag_{m}"]
        contrasts[f"{f}_interaction"] = contrasts[f"{f}_grad_grouping"] - contrasts[f"{f}_mag_grouping"]
    report = {
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "subjects": subjects, "seeds": list(SEEDS),
        "estimand": "Trial-weighted PCC; random seeds averaged within participant; participant bootstrap, 20000 draws, seed 0; exact two-sided sign flips; unadjusted exploratory tests",
        "cohort": {k: stats(v) for k, v in arrays.items()},
        "contrasts": {k: stats(v) for k, v in contrasts.items()},
        "per_participant": {k: v.tolist() for k, v in arrays.items()},
        "weighting_max_abs_cohort": max(abs(arrays[k].mean() - batch[k].mean()) for k in arrays),
        "weighting_max_abs_record_including_subsets": max(all_weighting),
        "seed_sensitivity": {},
    }
    for f, t in itertools.product(FILTERS, TAGS):
        v = (arrays[f"{f}_{t}_axis_partitioned"] - seed_arrays[f"{f}_{t}"]).mean(axis=1)
        report["seed_sensitivity"][f"{f}_{t}"] = dict(means=v.tolist(), minimum=float(v.min()), maximum=float(v.max()), sd=float(v.std(ddof=0)))
    return report, arrays, contrasts


def macro_name(key):
    return "Fx" + "".join(p.title() for p in key.split("_"))


def write():
    report, arrays, contrasts = load()
    data = json.loads(SOURCE.read_text())["per_subject"]
    decoding = {}
    for f, t, m in itertools.product(FILTERS, TAGS, MODES):
        records = [(data[sub][f][t]["random_partitions"]["42"] if m == "random" else data[sub][f][t][m]) for sub in report["subjects"]]
        decoding[f"{f}_{t}_{m}"] = {arm: stats(100*np.asarray([r["braille_5fold_bacc"][arm] for r in records]))
                                     for arm in ("pretrained_linear_probe", "control_a_linear_probe", "classical_bandpower")}
    report["decoding"] = decoding
    (BASE / "audit/factorial-results.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["% Generated by factorial_results.py. Do not transcribe values."]
    for family in ("cohort", "contrasts"):
        for k, v in report[family].items():
            n = macro_name(k)
            fmt = "+.4f" if family == "contrasts" else ".4f"
            for suffix, value in (("Mean", format(v["mean"], fmt)),
                                  ("CI", f'[{v["ci_low"]:{fmt}}, {v["ci_high"]:{fmt}}]'),
                                  ("Count", f'{v["positive"]}/{v["n"]}'),
                                  ("P", f'{v["p"]:.3f}')):
                lines.append("\\newcommand{\\" + n + suffix + "}{" + value + "}")
    for k, v in report["seed_sensitivity"].items():
        for field in ("minimum", "maximum", "sd"):
            lines.append("\\newcommand{\\" + macro_name(k) + field.title() + "}{" + f"{v[field]:.4f}" + "}")
    # Round the bound upwards so the prose remains a true bound.
    bound = np.ceil(report["weighting_max_abs_cohort"] * 10000) / 10000
    lines.append(r"\newcommand{\FxWeightingBound}{" + f"{bound:.4f}" + "}")
    for f, t in itertools.product(FILTERS, TAGS):
        cap = contrasts[f"{f}_{t}_mono_to_random"].mean()
        group = contrasts[f"{f}_{t}_grouping"].mean()
        for suffix, value in (("MonoToRandomPercent", 100*cap/(cap+group)), ("GroupingPercent", 100*group/(cap+group))):
            lines.append("\\newcommand{\\" + macro_name(f"{f}_{t}") + suffix + "}{" + f"{value:.1f}" + "}")
    probe_means = [v["pretrained_linear_probe"]["mean"] for v in decoding.values()]
    for suffix, value in (("Minimum", min(probe_means)), ("Maximum", max(probe_means))):
        lines.append("\\newcommand{\\FxProbe" + suffix + "}{" + f"{value:.2f}" + "}")
    for f in FILTERS:
        v = decoding[f"{f}_mag_monolithic"]["classical_bandpower"]
        for suffix, value in (("Mean", f'{v["mean"]:.2f}'), ("CI", f'[{v["ci_low"]:.2f}, {v["ci_high"]:.2f}]')):
            lines.append("\\newcommand{\\FxClassical" + f.title() + suffix + "}{" + value + "}")
    # Directional geometry counts share the same convention as supplement S2.
    single = json.loads((SOURCE.parent / "summary_phase3_ablations.json").read_text())["zero_shot_tokenizer_pcc_by_geometry_mode"]
    default = np.asarray(single["default"]["per_subject_pcc"])
    for mode, name in (("zero_orientation", "ZeroOrientation"), ("flip_orientation", "FlipOrientation"), ("scramble_channels", "ScrambleChannels")):
        delta = np.asarray(single[mode]["per_subject_pcc"]) - default
        count = int((delta < 0).sum())
        lines.append("\\newcommand{\\Sa" + name + "Count}{" + f"{count}/{len(delta)}" + "}")
    (BASE / "factorial_values.tex").write_text("\n".join(lines) + "\n")
    print("Generated factorial_values.tex and audit/factorial-results.json from 560 trial records.")
    return report, arrays, contrasts


if __name__ == "__main__":
    write()
