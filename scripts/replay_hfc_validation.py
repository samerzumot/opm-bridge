#!/usr/bin/env python3
"""Compare historical QR and corrected SVD HFC on separately replayed caches.

Results are representative within-subject diagnostics, not a cohort reproduction.
Run with PYTHONPATH=src:. and HF_HOME pointing at the pinned checkpoint cache.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from opm_bridge.preprocessing.sensor_mapping import compute_soft_hfc_projector, sensortype_wise_normalize
from opm_bridge.preprocessing.validation import geometry_checks, write_report


def legacy_projector(orientations, alpha=.25):
    q = np.asarray(orientations, dtype=np.float64)
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    valid = (norms > 1e-8).flatten()
    p = np.eye(len(q), dtype=np.float64)
    if valid.any():
        basis, _ = np.linalg.qr(q[valid] / np.maximum(norms[valid], 1e-8), mode='reduced')
        indices = np.flatnonzero(valid)
        p[np.ix_(indices, indices)] -= alpha * (basis @ basis.T)
    return p.astype(np.float32)


def difference(old, new):
    delta = new.astype(np.float64) - old.astype(np.float64)
    scale = np.linalg.norm(old.astype(np.float64))
    return dict(bitwise_equal=bool(np.array_equal(old, new)),
                max_abs=float(np.max(np.abs(delta))),
                relative_l2=float(np.linalg.norm(delta) / scale) if scale else None)


def reconstruction(model, x, pos, tags, groups, batch_size=16):
    from opm_bridge.models.vendor_brainomni.model_utils.loss import get_pcc
    from scripts.run_nottingham_tag_partition_factorial import compute_per_trial_pcc
    tok = model.tokenizer
    batch_scores, trials = [], []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start+batch_size]).float()
            pb = torch.from_numpy(pos).float()[None].expand(len(xb), -1, -1)
            sb = torch.from_numpy(tags).long()[None].expand(len(xb), -1)
            target = tok.norm_target(tok.unfold(xb, overlap_ratio=0.))
            reconstructed = torch.zeros_like(target)
            for group in groups:
                se = tok.sensor_embed(pb[:, group], sb[:, group])
                feat = tok.encoder(tok.unfold(xb[:, group], overlap_ratio=0.), se)
                quantized, _, _ = tok.quantizer(feat)
                reconstructed[:, group] = tok.decoder(quantized, se).float()
            batch_scores.append(float(get_pcc(target, reconstructed)))
            trials.extend(compute_per_trial_pcc(reconstructed, target).tolist())
    return dict(batch_weighted_pcc=float(np.mean(batch_scores)),
                trial_weighted_pcc=float(np.mean(trials)), per_trial_pcc=trials)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--face')
    parser.add_argument('--nottingham')
    parser.add_argument('--out', required=True)
    parser.add_argument('--skip-model', action='store_true')
    parser.add_argument('--tags', nargs='+', type=int, choices=[1, 2], default=[1, 2])
    parser.add_argument('--modes', nargs='+', choices=['monolithic', 'axis_partitioned', 'random_seed42'], default=['monolithic', 'axis_partitioned', 'random_seed42'])
    args = parser.parse_args()
    if not (args.face or args.nottingham):
        parser.error('Provide at least one replay cache')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = None
    if not args.skip_model:
        from opm_bridge.models.wrapper import load_brainomni_backbone
        model = load_brainomni_backbone().cpu().eval()
    report = dict(checkpoint=dict(repo='OpenTSLab/BrainOmni', revision='9a4d3c70495370397ccfbfd6d2496f25647545a5', size='tiny'), scope='Representative full recordings; historical cohort caches unavailable',
                  device='cpu', subjects={})
    for dataset, path in [('ds005107', args.face), ('nottingham', args.nottingham)]:
        if path is None:
            continue
        # Only locally created replay caches are loaded here.
        payload = torch.load(path, map_location='cpu', weights_only=False)
        pos = payload['pos_ori'].numpy()
        orientations = pos[:, 3:6]
        x = payload['x' if dataset == 'ds005107' else 'x_raw_ft'].numpy()
        old_p = legacy_projector(orientations)
        new_p = compute_soft_hfc_projector(orientations)
        old, new = old_p @ x.astype(np.float32), new_p @ x.astype(np.float32)
        if dataset == 'nottingham':
            tags = payload['sensor_type'].numpy()
            old = np.stack([sensortype_wise_normalize(v, tags) for v in old])
            new = np.stack([sensortype_wise_normalize(v, tags) for v in new])
        item = dict(subject=payload['subject_id'], shape=list(x.shape),
                    cache_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                    geometry=geometry_checks(orientations), operator_difference=difference(old_p, new_p),
                    model_input_difference=difference(old, new), reconstruction={})
        report['subjects'][dataset] = item
        write_report(args.out, report)
        groups = {'monolithic': [list(range(x.shape[1]))]}
        if dataset == 'nottingham':
            axes = payload['meta'].axis_labels
            groups['axis_partitioned'] = [[i for i, a in enumerate(axes) if a == axis] for axis in ('Z', 'X', 'Y')]
            permutation = np.random.default_rng(42).permutation(x.shape[1])
            groups['random_seed42'] = [g.tolist() for g in np.split(permutation, np.cumsum([len(g) for g in groups['axis_partitioned']])[:-1])]
        if model is not None:
            for tag in args.tags:
                for mode, partition in groups.items():
                    if mode not in args.modes:
                        continue
                    name = f'{tag}_{mode}'
                    print(dataset, name, 'old QR', flush=True)
                    tags = np.full(x.shape[1], tag, dtype=np.int64)
                    previous = reconstruction(model, old, pos, tags, partition)
                    if np.array_equal(old, new):
                        current = previous
                        method = 'Identical model input bytes; deterministic inference reused'
                    else:
                        print(dataset, name, 'new SVD', flush=True)
                        current = reconstruction(model, new, pos, tags, partition)
                        method = 'Separate old/new deterministic CPU inference'
                    item['reconstruction'][name] = dict(old=previous, new=current, comparison_method=method,
                        delta_pcc=current['trial_weighted_pcc']-previous['trial_weighted_pcc'])
                    write_report(args.out, report)
        print(dataset, item['operator_difference'], item['model_input_difference'], flush=True)
        write_report(args.out, report)


if __name__ == '__main__':
    main()
