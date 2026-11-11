"""Distinguish whole-record spectral changes from filter-edge effects."""
from pathlib import Path
import mne
import numpy as np
from opm_bridge.preprocessing.validation import SETTINGS,FILTER_KWARGS,spectral_summary,compare_spectra,write_report
p=Path('data/validation/face_raw/sub-01_ses-01_task-face_run-01_meg.fif')
raw=mne.io.read_raw_fif(p,preload=True,verbose=False)
raw.pick(mne.pick_types(raw.info,meg=True,ref_meg=False))
original=raw.get_data()
annotations=[dict(onset=float(o),duration=float(d),description=str(s)) for o,d,s in zip(raw.annotations.onset,raw.annotations.duration,raw.annotations.description)]
settings=SETTINGS['ds005107']
raw.notch_filter(settings['notch_hz'],**FILTER_KWARGS,verbose=False)
raw.filter(*settings['bandpass_hz'],**FILTER_KWARGS,verbose=False)
filtered=raw.get_data(); sf=raw.info['sfreq']
report=dict(annotations=annotations,segments={})
for trim in [0,5,10,20]:
 start=round(trim*sf); stop=original.shape[1]-start
 before=spectral_summary(original[:,start:stop],sf,raw.ch_names,'T')
 after=spectral_summary(filtered[:,start:stop],sf,raw.ch_names,'T')
 shield=np.array(compare_spectra(before,after)['shielding_db'],float)
 f=np.array(before['frequencies_hz'])
 report['segments'][str(trim)]=dict(trim_seconds_each_end=trim,
    median_shielding_db={str(hz):float(np.median(shield[:,np.argmin(abs(f-hz))])) for hz in [1,2,3,10,44,50,100]},
    std_before=float(np.median(np.std(original[:,start:stop],axis=-1))),
    std_after=float(np.median(np.std(filtered[:,start:stop],axis=-1))))
import json
qc=json.loads(next(Path('data/cache/validation/ds005107').rglob('*.qc.json')).read_text())
events=qc['input_event_samples']; duration=original.shape[1]/sf
report['epoch_overlap_check']=dict(edge_seconds=5, first_event_seconds=min(events)/sf, last_event_seconds=max(events)/sf, recording_duration_seconds=duration, overlapping_epoch_indices=[i for i,v in enumerate(events) if v/sf-.2<5 or v/sf+1.8>duration-5])
write_report('reviews/preprocessing-2026-11-11/face_filter_edge_check.json',report)
print(report,flush=True)
