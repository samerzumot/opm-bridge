#!/usr/bin/env python3
"""Compare aligned physical-unit stages exported as NPZ, without pickle loading."""
import argparse

import numpy as np

from opm_bridge.preprocessing.validation import (
    TUTORIAL, compare_spectra, epoch_checks, geometry_checks, sha256_bytes,
    spectral_summary, write_report,
)
from opm_bridge.preprocessing.sensor_mapping import apply_soft_hfc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="NPZ: before, after, sfreq, ch_names, unit; optional orientations, labels, tmin")
    parser.add_argument("--out", required=True, help="Output JSON report")
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        sfreq = float(data["sfreq"])
        names, unit = data["ch_names"].tolist(), str(data["unit"].item())
        before = spectral_summary(data["before"], sfreq, names, unit)
        after = spectral_summary(data["after"], sfreq, names, unit)
        report = dict(tutorial=TUTORIAL, status="requires_scientific_review",
                      before=before, after=after, comparison=compare_spectra(before, after))
        if "orientations" in data:
            report["geometry"] = geometry_checks(data["orientations"])
            soft = apply_soft_hfc(data["before"], data["orientations"], .25)
            soft_summary = spectral_summary(soft, sfreq, names, unit)
            report["soft_hfc_comparison"] = compare_spectra(before, soft_summary)
            report["soft_hfc_after"] = soft_summary
        if data["after"].ndim == 3 and "labels" in data:
            report["epochs"] = epoch_checks(data["after"], data["labels"], sfreq,
                                            float(data["tmin"]) if "tmin" in data else -.2)
            # Save per-channel evoked traces to inspect amplitude/latency changes.
            report["evoked_by_label"] = {
                str(label): {key: data[key][data["labels"] == label].mean(axis=0).tolist()
                             for key in ("before", "after")}
                for label in np.unique(data["labels"])
            }
    from pathlib import Path
    report["input_sha256"] = sha256_bytes(Path(args.input).read_bytes())
    write_report(args.out, report)
    print(f"Diagnostics saved to {args.out}; scientific validation remains pending.")


if __name__ == "__main__":
    main()
