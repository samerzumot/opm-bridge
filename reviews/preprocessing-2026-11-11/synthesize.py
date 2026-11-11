"""Assemble dated, evidence-backed findings without modifying historical results."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from opm_bridge.preprocessing.validation import write_report
root=Path('reviews/preprocessing-2026-11-11')
load=lambda name: json.loads((root/name).read_text())
face=load('face_hfc_comparison.json')['subjects']['ds005107']
nottingham=load('nottingham_hfc_comparison.json')['subjects']['nottingham']
history_path=Path('results/summary_phase4_nottingham_factorial_and_seeds.json')
historical=json.loads(history_path.read_text())['per_subject']['sub-101']['softhfc']['mag']
comparisons={}
for name,result in nottingham['reconstruction'].items():
    mode=name.split('_',1)[1]
    previous=historical[mode]
    comparisons[mode]=dict(replay_pcc=result['new']['trial_weighted_pcc'],
        historical_pcc=previous['pcc_trial_weighted'],
        replay_minus_historical=result['new']['trial_weighted_pcc']-previous['pcc_trial_weighted'],
        max_abs_per_trial_difference=float(np.max(np.abs(np.array(result['new']['per_trial_pcc'])-np.array(previous['trial_pccs'])))),
        old_qr_vs_new_svd_delta=result['delta_pcc'])
checkpoint=Path('data/validation/huggingface/hub/models--OpenTSLab--BrainOmni/snapshots/9a4d3c70495370397ccfbfd6d2496f25647545a5/tiny/BrainOmni.pt')
report=dict(date='2026-11-11',
 verdict='No HFC-fix effect in representative replays; Nottingham source cohort operators identical; event-index issue remains',
 versions={p:importlib.metadata.version(p) for p in ['mne','numpy','scipy','torch']},
 checkpoint=dict(repo='OpenTSLab/BrainOmni',revision='9a4d3c70495370397ccfbfd6d2496f25647545a5',sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()),
 representatives={'FACE-DEC':dict(subject='sub-01',run='run-01',trials=256,channels=60,
   operator_difference=face['operator_difference'],model_input_difference=face['model_input_difference'],
   pcc_by_tag={k:dict(old=v['old']['trial_weighted_pcc'],new=v['new']['trial_weighted_pcc'],delta=v['delta_pcc']) for k,v in face['reconstruction'].items()},
   historical_comparison='Not an exact reproduction of the reported 21-subject, 50-channel aligned analysis; representative replay retains 60 channels'),
   'Nottingham':dict(subject='sub-101',trials=84,channels=176,classes=[42,42],
   source_zip_crc32='ee310249',source_uncompressed_bytes=579156480,
   operator_difference=nottingham['operator_difference'],model_input_difference=nottingham['model_input_difference'],historical_comparisons=comparisons)},
 nottingham_cohort_geometry=load('nottingham_cohort_geometry.json'),
 timing_issue=load('raw_replay_qc_summary.json')['Nottingham']['timing_check'],
 face_edge_check=load('face_filter_edge_check.json'),
 test_result=dict(passed=23,failed=0,environment='Project .venv with global pytest appended, plugin autoload disabled'),
 limitations=[
   'Full 21-subject FACE-DEC alignment, decoding and cohort confidence intervals were not rerun.',
   'Only sub-101 raw data and two MAG reconstruction modes were replayed for Nottingham; all ten source geometries were checked.',
   'Identical float32 HFC operators imply unchanged downstream outputs for matching inputs and deterministic execution. Historical geometry bytes were not archived; current source retained-channel counts match every participant.',
   'Zero old/new input differences allowed deterministic reconstruction to be reused instead of redundant identical inference.',
   'Positive shielding and visible evoked responses do not prove neural-signal preservation; Soft-HFC changes condition-average waveforms.',
   'Nottingham event samples are one greater than rounded onset times 1200 for all 84 stimulus events. This replay preserves historical indexing to isolate the HFC change.',
   'The source Nottingham MEG JSON reports DigitizedLandmarks=false, while the adapter sets has_digitized_landmarks=True. This metadata flag needs correction or justification; it is not consumed by the HFC operator.'
 ],
 historical_results_sha256=hashlib.sha256(history_path.read_bytes()).hexdigest())
write_report(root/'findings.json',report)
print(json.dumps(comparisons,indent=2))
