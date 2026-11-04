#!/usr/bin/env python3
"""Zero-disk HTTP Range streamer for Nottingham Triaxial OPM-MEG (Zenodo 11126593, Adults.zip).

Streams coregistered triaxial (`X/Y/Z`, `CoordinateFrame.HEAD`) QuSpin QZFM recordings for adult
subjects (`sub-101`..`sub-110`) directly from Zenodo's `Adults.zip` archive using HTTP Range
requests without downloading the 12.24 GB archive to disk.
"""

from __future__ import annotations

import argparse
import io
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import List

import mne
import numpy as np
import torch

from opm_bridge.data.nottingham_opm import (
    extract_nottingham_sensor_metadata,
    parse_meg_read_info,
    parse_nottingham_braille_events,
)
from opm_bridge.preprocessing.sensor_mapping import normalize_sensor_positions

ZENODO_ADULTS_ZIP_URL = "https://zenodo.org/api/records/11126593/files/Adults.zip/content"


class HttpRangeFile(io.RawIOBase):
    """Seekable HTTP Range reader for streaming individual entries from remote ZIP archives."""

    def __init__(self, url: str) -> None:
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def readinto(self, b: bytearray | memoryview) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b) - 1, self.size - 1)
        data = b""
        for attempt in range(5):
            try:
                req = urllib.request.Request(
                    self.url, headers={"Range": f"bytes={self.pos}-{end}"}
                )
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(1.5 * (attempt + 1))
        n = len(data)
        b[:n] = data
        self.pos += n
        return n


def stream_nottingham_subjects(
    subjects: List[str],
    out_dir: str | Path = "data/cache/nottingham",
    stream_seconds: int | None = None,
    overwrite: bool = False,
) -> List[Path]:
    """Stream and preprocess adult triaxial OPM subjects from Zenodo 11126593."""
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    pending = [
        s
        for s in subjects
        if overwrite or not (out_dir_p / f"{s}_ses-001_task-braille_run-001_256hz.pt").exists()
    ]
    if not pending:
        for s in subjects:
            p = out_dir_p / f"{s}_ses-001_task-braille_run-001_256hz.pt"
            print(f"Already cached: {p}")
            saved.append(p)
        return saved

    f = io.BufferedReader(HttpRangeFile(ZENODO_ADULTS_ZIP_URL), buffer_size=4 * 1024 * 1024)
    with zipfile.ZipFile(f) as zf:
        all_names = set(zf.namelist())
        for sub in subjects:
            out_pt = out_dir_p / f"{sub}_ses-001_task-braille_run-001_256hz.pt"
            if out_pt.exists() and not overwrite:
                print(f"Already cached: {out_pt}")
                saved.append(out_pt)
                continue

            prefix = f"Adults/{sub}/ses-001/meg/{sub}_ses-001_task-braille_run-001"
            if f"{prefix}_meg.dat" not in all_names:
                print(f"Skipping {sub}: {prefix}_meg.dat not found in Adults.zip")
                continue

            t0 = time.time()
            with tempfile.TemporaryDirectory() as tmp:
                tmp_p = Path(tmp)
                ch_p = tmp_p / "channels_proc.tsv"
                ev_p = tmp_p / "events.tsv"
                ri_p = tmp_p / "read_info.txt"
                ch_p.write_bytes(zf.read(f"{prefix}_channels_proc.tsv"))
                ev_p.write_bytes(zf.read(f"{prefix}_events.tsv"))
                ri_p.write_bytes(zf.read(f"{prefix}_meg_read_info.txt"))

                meta, good_idx = extract_nottingham_sensor_metadata(ch_p)
                ev_dict = parse_nottingham_braille_events(ev_p)
                (n_rows, n_cols), dt, _ = parse_meg_read_info(ri_p)

                if stream_seconds is not None:
                    n_samp_stream = min(n_cols, int(1200 * stream_seconds))
                else:
                    n_samp_stream = n_cols
                n_bytes = n_rows * n_samp_stream * 8
                info = zf.getinfo(f"{prefix}_meg.dat")
                with zf.open(info) as zstream:
                    raw_bytes = zstream.read(n_bytes)

                raw_1d = np.frombuffer(raw_bytes, dtype=dt)
                n_actual_cols = len(raw_1d) // n_rows
                mat = (
                    raw_1d[: n_rows * n_actual_cols]
                    .reshape((n_rows, n_actual_cols), order="F")
                    .astype(np.float64)
                )
                mat_good = mat[good_idx, :]  # (C_good, n_samples) in fT

                mat_filt = mne.filter.notch_filter(
                    mat_good, Fs=1200.0, freqs=[50.0, 100.0], verbose=False
                )
                mat_filt = mne.filter.filter_data(
                    mat_filt, sfreq=1200.0, l_freq=1.0, h_freq=45.0, verbose=False
                )

                ep_list, y_list = [], []
                pre_s, post_s = int(0.2 * 1200), int(1.8 * 1200)
                for lbl, samps in [
                    (0, ev_dict["index_samples"]),
                    (1, ev_dict["pinky_samples"]),
                ]:
                    for s_idx in samps:
                        if s_idx - pre_s >= 0 and s_idx + post_s <= n_actual_cols:
                            ep = mat_filt[:, s_idx - pre_s : s_idx + post_s]
                            ep = ep - ep[:, :pre_s].mean(axis=1, keepdims=True)
                            ep_list.append(ep)
                            y_list.append(lbl)

                ep_arr = np.stack(ep_list, axis=0)  # (N_trials, C_good, 2400)
                ep_256 = mne.filter.resample(
                    ep_arr, down=1200.0 / 256.0, npad="auto", axis=-1
                )[:, :, :512].astype(np.float32)

                pos_def, st_def, _ = normalize_sensor_positions(meta, "default")
                torch.save(
                    {
                        "subject_id": sub,
                        "x_raw_ft": torch.from_numpy(ep_256),
                        "labels": torch.tensor(y_list, dtype=torch.long),
                        "meta": meta,
                        "pos_ori": torch.from_numpy(pos_def),
                        "sensor_type": torch.from_numpy(st_def),
                    },
                    out_pt,
                )
                saved.append(out_pt)
                print(
                    f"Cached {sub} in {time.time() - t0:.1f}s: "
                    f"shape={ep_256.shape}, labels={np.bincount(y_list).tolist()}",
                    flush=True,
                )
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream Nottingham triaxial OPM subjects from Zenodo")
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=[f"sub-{i}" for i in range(101, 127)],
        help="Adult subjects to stream (default: sub-101..sub-126)",
    )
    parser.add_argument("--out-dir", type=str, default="data/cache/nottingham")
    parser.add_argument(
        "--stream-seconds",
        type=int,
        default=None,
        help="Optional limit on recording seconds to stream (default: None = full recording)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cached files")
    args = parser.parse_args()

    print(f"Streaming {len(args.subjects)} subjects from Zenodo 11126593...")
    paths = stream_nottingham_subjects(
        args.subjects,
        out_dir=args.out_dir,
        stream_seconds=args.stream_seconds,
        overwrite=args.overwrite,
    )
    print(f"Completed: cached {len(paths)} subjects in {args.out_dir}")


if __name__ == "__main__":
    main()

