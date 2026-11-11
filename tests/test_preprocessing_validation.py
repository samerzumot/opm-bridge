"""Preprocessing regression checks independent of model weights and recordings."""

import numpy as np
import pytest

from opm_bridge.preprocessing.sensor_mapping import compute_soft_hfc_projector


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("alpha", [0.25, 1.0])
def test_hfc_preserves_complement_for_deficient_geometry(rank, alpha):
    rng = np.random.default_rng(7)
    q = rng.normal(size=(8, 3))
    q[:, rank:] = 0
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    expected = np.eye(8) - alpha * q @ np.linalg.pinv(q)
    actual = compute_soft_hfc_projector(q, alpha)
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    complement = (np.eye(8) - q @ np.linalg.pinv(q)) @ rng.normal(size=(8, 20))
    np.testing.assert_allclose(actual @ complement, complement, atol=2e-7)


def test_hfc_zero_rows_and_invalid_metadata():
    q = np.array([[1., 0., 0.], [0., 0., 0.], [1., 0., 0.]])
    p = compute_soft_hfc_projector(q)
    np.testing.assert_array_equal(p[1], [0., 1., 0.])
    np.testing.assert_array_equal(compute_soft_hfc_projector(np.zeros((4, 3))), np.eye(4))
    for bad in (np.ones((4, 4)), np.full((4, 3), np.nan)):
        with pytest.raises(ValueError):
            compute_soft_hfc_projector(bad)


from pathlib import Path
import importlib.util
import json

from opm_bridge.preprocessing.validation import (
    SETTINGS, compare_spectra, epoch_checks, geometry_checks, spectral_summary, write_report,
)


def test_spectral_shielding_units_and_flat_channels(tmp_path):
    rng = np.random.default_rng(22)
    x = rng.normal(size=(3, 1024))
    before = spectral_summary(x, 256., ['a', 'b', 'c'], 'fT')
    after = spectral_summary(x * .75, 256., ['a', 'b', 'c'], 'fT')
    result = compare_spectra(before, after)
    np.testing.assert_allclose(result['shielding_db'], 10 * np.log10(1 / .75**2))
    with pytest.raises(ValueError, match='unit'):
        compare_spectra(before, {**after, 'unit': 'T'})
    with pytest.raises(ValueError, match='ch_names'):
        compare_spectra(before, {**after, 'ch_names': ['b', 'a', 'c']})
    flat = spectral_summary(np.zeros((1, 512)), 256., ['flat'], 'T')
    assert flat['flat_channels'] == ['flat']
    report = compare_spectra(flat, flat)
    assert report['undefined_bins'] == 129
    write_report(tmp_path / 'flat.json', report)
    assert 'NaN' not in (tmp_path / 'flat.json').read_text()
    with pytest.raises(ValueError):
        spectral_summary(x * np.nan, 256., ['a', 'b', 'c'], 'T')
    with pytest.raises(ValueError):
        spectral_summary(x, 256., ['a', 'b', 'c'], 'normalized')


def test_epoch_labels_and_geometry_checks():
    x = np.zeros((2, 3, 512))
    assert epoch_checks(x, [0, 1])['event_offset_samples'] == 51.2
    for bad, labels in [(x[..., :511], [0, 1]), (x, [0]), (x[:0], [])]:
        with pytest.raises(ValueError):
            epoch_checks(bad, labels)
    assert geometry_checks(np.tile([1., 0., 0.], (5, 1)))['rank'] == 1


def test_yaml_matches_runtime_settings():
    import yaml
    root = Path(__file__).resolve().parents[1]
    for dataset, filename in [('ds005107', 'ds005107_primary.yaml'), ('nottingham_adults', 'nottingham_adults.yaml')]:
        spec = yaml.safe_load((root / 'configs' / filename).read_text())['dataset']
        for key, value in SETTINGS[dataset].items():
            assert spec[key] == value


def _script(name):
    path = Path(__file__).resolve().parents[1] / 'scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_face_cache_qc_and_legacy_rejection(tmp_path, monkeypatch):
    import mne
    import torch
    module = _script('stream_and_cache')
    info = mne.create_info(['a', 'b', 'c', 'STIM'], 1000., ['mag'] * 3 + ['stim'])
    for ch in info['chs'][:3]:
        ch['loc'][:3] = [.01, .02, .03]
        ch['loc'][9:12] = [1., 0., 0.]
    x = np.random.default_rng(8).normal(scale=1e-12, size=(4, 12000))
    x[3] = 0
    x[3, 3000:3010] = 1
    raw = mne.io.RawArray(x, info, verbose=False)
    path = tmp_path / 'test_raw.fif'
    raw.save(path, overwrite=True, verbose=False)
    monkeypatch.setattr(module, '_download_bytes', lambda url: path.read_bytes() if url.endswith('.fif') else b'events')
    monkeypatch.setattr(module, 'align_ds005107_events', lambda *a, **kw: np.array([[3033, 0, 1]]))
    output = module.stream_and_cache_ds005107_run('test', out_dir=tmp_path / 'cache', use_soft_hfc=True)
    qc = json.loads(output.with_suffix('.qc.json').read_text())
    assert qc['geometry']['rank'] == 1
    assert qc['epochs']['samples'] == 512
    assert qc['retained_event_indices'] == [0]
    assert len(qc['provenance']['sources']) == 2
    assert qc['outlier_rejection_applied'] is False
    assert torch.load(output, weights_only=True)['x'].shape == (1, 3, 512)
    assert module.stream_and_cache_ds005107_run('test', out_dir=tmp_path / 'cache', use_soft_hfc=True) == output
    with pytest.raises(ValueError, match='mismatched'):
        module.stream_and_cache_ds005107_run('test', out_dir=tmp_path / 'cache', use_soft_hfc=True, hfc_alpha=.5)
    output.with_suffix('.qc.json').unlink()
    with pytest.raises(ValueError, match='Legacy'):
        module.stream_and_cache_ds005107_run('test', out_dir=tmp_path / 'cache', use_soft_hfc=True)


def test_nottingham_cache_uses_declared_rate_and_dtype(tmp_path, monkeypatch):
    import zipfile
    module = _script('stream_nottingham_zenodo')
    prefix = 'Adults/sub-101/ses-001/meg/sub-101_ses-001_task-braille_run-001'
    archive = tmp_path / 'Adults.zip'
    x = np.random.default_rng(5).normal(size=(3, 12000)).astype('<f4')
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr(prefix + '_meg.dat', x.tobytes(order='F'))
        z.writestr(prefix + '_meg_read_info.txt', 'Size 3 x 12000\nPrecision single\nOrdering ieee-le\n')
        z.writestr(prefix + '_meg.json', json.dumps({'SamplingFrequency': 1000.}))
        z.writestr(prefix + '_events.tsv', 'type\tsample\tonset\nStart_index\t3001\t3\nStart_pinky\t7001\t7\nStart_pinky\t12000\t11.999\n')
        z.writestr(prefix + '_channels_proc.tsv', 'name\tstatus\tPx\tPy\tPz\tOx\tOy\tOz\nA_X\tgood\t5\t2\t6\t1\t0\t0\nA_Y\tgood\t5\t2\t6\t0\t1\t0\nA_Z\tbad\t5\t2\t6\t0\t0\t1\n')
    monkeypatch.setattr(module, 'HttpRangeFile', lambda url: open(archive, 'rb', buffering=0))
    output = module.stream_nottingham_subjects(['sub-101'], out_dir=tmp_path / 'cache')[0]
    qc = json.loads(output.with_suffix('.qc.json').read_text())
    assert qc['provenance']['sfreq_raw'] == 1000.
    assert qc['epochs']['samples'] == 512
    assert qc['epochs']['labels'] == {'0': 1, '1': 1}
    assert qc['excluded_channel_indices'] == [2]
    assert qc['retained_events'] == [dict(sample=3000, label=0), dict(sample=7000, label=1)]
    assert qc['provenance']['event_sample_offset'] == -1
    assert qc['provenance']['event_onset_validation'] == 'passed'
    assert qc['dropped_events'] == [dict(sample=11999, label=1, reason='outside_recording')]
    assert module.stream_nottingham_subjects(['sub-101'], out_dir=tmp_path / 'cache') == [output]


def test_validation_cli_exports_evoked(tmp_path, monkeypatch):
    import sys
    module = _script('validate_preprocessing')
    x = np.random.default_rng(4).normal(size=(4, 3, 512))
    source, out = tmp_path / 'stages.npz', tmp_path / 'report.json'
    np.savez(source, before=x, after=.75*x, sfreq=256., ch_names=['a', 'b', 'c'],
             unit='fT', orientations=np.eye(3), labels=[0, 1, 0, 1], tmin=-.2)
    monkeypatch.setattr(sys, 'argv', ['validate_preprocessing', str(source), '--out', str(out)])
    module.main()
    report = json.loads(out.read_text())
    assert report['status'] == 'requires_scientific_review'
    assert set(report['evoked_by_label']) == {'0', '1'}
    assert len(report['input_sha256']) == 64


def test_hard_hfc_matches_mne_order_one():
    import mne
    rng = np.random.default_rng(17)
    info = mne.create_info([f'OPM{i}' for i in range(12)], 256., 'mag')
    orientations = []
    for ch in info['chs']:
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        ch['loc'][:3] = rng.normal(size=3) * .05
        ch['loc'][3:] = rotation.T.ravel()
        ch['coil_type'] = 2000
        orientations.append(rotation[:, 2])
    x = rng.normal(size=(12, 512))
    raw = mne.io.RawArray(x, info, verbose=False)
    raw.add_proj(mne.preprocessing.compute_proj_hfc(info, order=1, verbose=False))
    raw.apply_proj(verbose=False)
    np.testing.assert_allclose(compute_soft_hfc_projector(orientations, 1.) @ x,
                               raw.get_data(), atol=2e-7)


def test_near_deficient_basis_and_full_rank_legacy_equivalence():
    rng = np.random.default_rng(18)
    q = rng.normal(size=(15, 3))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    basis, _ = np.linalg.qr(q, mode='reduced')
    np.testing.assert_allclose(compute_soft_hfc_projector(q), np.eye(15)-.25*basis@basis.T, atol=1e-7)
    q[:, 2] = 1e-17 * q[:, 0]
    p = compute_soft_hfc_projector(q, 1.)
    assert np.linalg.matrix_rank(p) == 13


def test_nottingham_one_based_samples_are_zero_based(tmp_path):
    from opm_bridge.data.nottingham_opm import parse_nottingham_braille_events
    p = tmp_path / 'events.tsv'
    p.write_text('onset\tsample\ttype\n0\t1\tStart_index\n2.7675\t3322\tStart_index\n6.44666666666667\t7737\tStart_pinky\n')
    events = parse_nottingham_braille_events(p)
    np.testing.assert_array_equal(events['index_samples'], [0, 3321])
    np.testing.assert_array_equal(events['pinky_samples'], [7736])


@pytest.mark.parametrize('sample,onset', [(0, 0), (1.5, 0), (2, 0), ('nan', 0), (1, 'nan')])
def test_nottingham_rejects_invalid_event_convention(tmp_path, sample, onset):
    from opm_bridge.data.nottingham_opm import parse_nottingham_braille_events
    path = tmp_path / 'events.tsv'
    path.write_text(f'onset\tsample\ttype\n{onset}\t{sample}\tStart_index\n')
    with pytest.raises(ValueError):
        parse_nottingham_braille_events(path, sfreq=1200.)


def test_nottingham_old_timing_cache_requires_rebuild(tmp_path):
    from opm_bridge.preprocessing.validation import PIPELINE_VERSION
    module = _script('stream_nottingham_zenodo')
    path = tmp_path / 'sub-101_ses-001_task-braille_run-001_256hz.pt'
    path.touch()
    path.with_suffix('.qc.json').write_text(json.dumps({'provenance': {
        'pipeline_version': PIPELINE_VERSION, 'stream_seconds': None}}))
    with pytest.raises(ValueError, match='Legacy or mismatched'):
        module.stream_nottingham_subjects(['sub-101'], out_dir=tmp_path)
