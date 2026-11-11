"""Audit source metadata for all ten reported Nottingham participants."""
import io
import json
import zipfile
from pathlib import Path
from scripts.stream_nottingham_zenodo import HttpRangeFile,ZENODO_ADULTS_ZIP_URL
from scripts.replay_hfc_validation import legacy_projector,difference
from opm_bridge.data.nottingham_opm import extract_nottingham_sensor_metadata
from opm_bridge.preprocessing.sensor_mapping import compute_soft_hfc_projector
from opm_bridge.preprocessing.validation import geometry_checks,write_report,sha256_bytes
root=Path('data/validation/source_metadata');root.mkdir(parents=True,exist_ok=True)
records={}
historical=json.loads(Path('results/summary_phase4_nottingham_factorial_and_seeds.json').read_text())['per_subject']
with zipfile.ZipFile(io.BufferedReader(HttpRangeFile(ZENODO_ADULTS_ZIP_URL),buffer_size=65536)) as z:
    for number in range(101,111):
        sub=f'sub-{number}'
        name=f'{sub}_ses-001_task-braille_run-001_channels_proc.tsv'
        path=root/name
        if not path.exists():
            path.write_bytes(z.read(f'Adults/{sub}/ses-001/meg/{name}'))
        meta,_=extract_nottingham_sensor_metadata(path)
        record=dict(geometry=geometry_checks(meta.ori_3d),n_channels=len(meta.ch_names),
                    reported_n_channels=historical[sub]['raw']['mag']['monolithic']['n_channels'],
                    source_sha256=sha256_bytes(path.read_bytes()),
                    operator_difference=difference(legacy_projector(meta.ori_3d),compute_soft_hfc_projector(meta.ori_3d)))
        records[sub]=record
        print(sub,record,flush=True)
write_report('reviews/preprocessing-2026-11-11/nottingham_cohort_geometry.json',records)
