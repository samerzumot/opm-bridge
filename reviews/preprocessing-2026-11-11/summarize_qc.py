import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from opm_bridge.preprocessing.validation import spectral_summary,compare_spectra,write_report
from opm_bridge.preprocessing.sensor_mapping import apply_soft_hfc
import torch
out=Path('reviews/preprocessing-2026-11-11')
summary={}
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for row,(name,folder) in enumerate([('FACE-DEC','ds005107'),('Nottingham','nottingham')]):
 p=next(Path('data/cache/validation',folder).rglob('*.qc.json'))
 d=json.loads(p.read_text())
 before=d.get('temporal_before',d.get('before')); after=d.get('temporal_after',d.get('after'))
 f=np.array(before['frequencies_hz']); pre=np.array(before['psd']); post=np.array(after['psd'])
 shielding=10*np.log10(pre/post)
 item=dict(geometry=d['geometry'],epochs=d['epochs'],versions=d['provenance']['versions'],
  unit=before['unit'],raw_shape=before['shape'],sources=d['provenance']['sources'],
  retained_channels=d['retained_channels'],flat_channels_before=before['flat_channels'],flat_channels_after=after['flat_channels'],
  spectral_median_shielding_db={})
 for label,lo,hi in [('drift_0_2',0,2),('passband_5_30',5,30),('line_49_51',49,51),('opm_43_45',43,45)]:
  item['spectral_median_shielding_db'][label]=float(np.median(shielding[:,(f>=lo)&(f<=hi)]))
 if name=='FACE-DEC':
  item.update(rejected_channels=d['flagged_outlier_channels'],input_events=len(d['input_event_samples']),retained_events=len(d['retained_event_indices']),reference_channels=d['reference_channels_excluded'])
  hfcpre,hfcpost=d['hfc_before'],d['hfc_after']
 else:
  payload=torch.load(p.with_suffix('').with_suffix('.pt'),weights_only=False)
  x=payload['x_raw_ft'].numpy(); ori=payload['pos_ori'].numpy()[:,3:]
  hfcpre=spectral_summary(x,256.,d['retained_channels'],'fT')
  hfcpost=spectral_summary(apply_soft_hfc(x,ori),256.,d['retained_channels'],'fT')
  item.update(excluded_channel_indices=d['excluded_channel_indices'],dropped_events=d['dropped_events'])
 hfc=compare_spectra(hfcpre,hfcpost)
 item['hfc_median_shielding_db']=float(np.median(np.array(hfc['shielding_db'],dtype=float)))
 summary[name]=item
 for y,label in [(pre,'Before temporal filters'),(post,'After temporal filters')]:
  axes[row,0].semilogy(f,np.median(y,axis=0),label=label)
 axes[row,0].set(xlim=(0,110),title=f'{name}: continuous spectra',xlabel='Frequency (Hz)',ylabel=f"PSD ({before['unit']}²/Hz)")
 axes[row,0].legend()
 axes[row,1].plot(f,np.median(shielding,axis=0),label='Temporal filters')
 hf=np.array(hfcpre['frequencies_hz'])
 axes[row,1].plot(hf,np.median(np.array(hfc['shielding_db'],dtype=float),axis=0),label='Soft-HFC on epochs')
 axes[row,1].set(xlim=(0,110),title=f'{name}: median shielding',xlabel='Frequency (Hz)',ylabel='Power reduction (dB)')
 axes[row,1].legend(); axes[row,1].axhline(0,color='gray',lw=.5)
events=pd.read_csv(next(Path('data/validation/source_metadata').glob('*events.tsv')),sep='\t')
selected=events[events.type.str.startswith(('Start_index','Start_pinky'),na=False)]
offsets=selected['sample'].to_numpy()-np.rint(selected.onset.to_numpy()*1200)
summary['Nottingham']['timing_check']=dict(stimulus_count=len(selected),sample_minus_rounded_onset_times_sfreq=np.unique(offsets).tolist(),implemented_epoch_shift_sec=1/1200,status='one_sample_late_relative_to_onset_column')
summary['verdict']='Representative processing completed; timing discrepancy requires resolution; power reduction alone does not validate neural preservation'
write_report(out/'raw_replay_qc_summary.json',summary)
fig.savefig(out/'raw_replay_spectra.png',dpi=160)
print(json.dumps({k:{key:v for key,v in item.items() if key in ['raw_shape','epochs','geometry','spectral_median_shielding_db','hfc_median_shielding_db','timing_check']} for k,item in summary.items() if isinstance(item,dict)},indent=2))
