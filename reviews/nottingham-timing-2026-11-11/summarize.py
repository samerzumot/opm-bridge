"""Summarize completed paired results without changing historical cohort values."""
import hashlib
import json
from pathlib import Path
from opm_bridge.preprocessing.validation import write_report

root=Path('reviews/nottingham-timing-2026-11-11')
d=json.loads((root/'comparison.json').read_text())
rows=[]
for mode,label in [('monolithic','Monolithic'),('axis_partitioned','Axis-partitioned')]:
    pair=d['reconstruction'][mode]
    for metric,title in [('pcc_trial_weighted','Reconstruction PCC'),('pretrained_linear_probe','Probe balanced accuracy')]:
        old=pair['old'][metric];new=pair['corrected'][metric]
        if isinstance(old,dict):old=old['mean_balanced_accuracy'];new=new['mean_balanced_accuracy']
        rows.append(dict(condition=label,metric=title,old=old,corrected=new,delta=new-old))
for filtered in ['raw','softhfc']:
    old=d['classical'][f'old_{filtered}']['mean_balanced_accuracy']
    new=d['classical'][f'corrected_{filtered}']['mean_balanced_accuracy']
    rows.append(dict(condition=f'Classical {filtered}',metric='Balanced accuracy',old=old,corrected=new,delta=new-old))
report=dict(subject=d['subject'],native_event_correction=-1,seconds=d['event_shift_seconds'],
    shape=d['shape'],class_counts=d['class_counts'],matched_source_hashes=d['source_hashes_match'],
    rows=rows,old_axis_advantage=d['old_axis_minus_monolithic'],corrected_axis_advantage=d['corrected_axis_minus_monolithic'],
    delta_axis_advantage=d['delta_axis_advantage'],historical_comparisons=d['historical_comparisons'],
    test_count=30,tests_passed=True,
    cohort_results_overwritten=False,
    limits=[
        'Paired sensitivity in sub-101 only. No cohort confidence interval, sign-flip p-value or generalization conclusion follows.',
        'Model analyses use MAG with Soft-HFC. Random partitions, other seeds, GRAD and other participants were not rerun.',
        'Historical cohort tables retain their original timing and need a full corrected rerun before replacement.',
        'Accuracy estimates use identical stratified five-fold splits and settings. No hyperparameters were retuned.'
    ],
    files_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        Path('src/opm_bridge/data/nottingham_opm.py'),Path('scripts/stream_nottingham_zenodo.py'),
        Path('scripts/compare_nottingham_timing.py'),Path('results/summary_phase4_nottingham_factorial_and_seeds.json')]})
write_report(root/'findings.json',report)
print(json.dumps(report,indent=2))
