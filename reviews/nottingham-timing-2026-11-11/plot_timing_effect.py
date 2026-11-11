"""Compare matched physical-unit evoked responses after the timing correction."""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from opm_bridge.preprocessing.sensor_mapping import apply_soft_hfc
from opm_bridge.preprocessing.validation import write_report

root=Path('reviews/nottingham-timing-2026-11-11')
old=torch.load('data/cache/validation/nottingham/sub-101_ses-001_task-braille_run-001_256hz.pt',weights_only=False)
new=torch.load('data/cache/validation/nottingham_timing_corrected/sub-101_ses-001_task-braille_run-001_256hz.pt',weights_only=False)
y=new['labels'].numpy();ori=new['pos_ori'].numpy()[:,3:]
a=apply_soft_hfc(old['x_raw_ft'].numpy(),ori);b=apply_soft_hfc(new['x_raw_ft'].numpy(),ori)
t=-.2+np.arange(512)/256
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
report={}
for col,(label,name) in enumerate([(0,'Index'),(1,'Pinky')]):
    before=a[y==label].mean(axis=0);after=b[y==label].mean(axis=0)
    old_rms=np.sqrt(np.mean(before**2,axis=0));new_rms=np.sqrt(np.mean(after**2,axis=0))
    change=np.sqrt(np.mean((after-before)**2,axis=0))
    axes[0,col].plot(t,old_rms,label='Historical timing',ls='--',color='#2563eb')
    axes[0,col].plot(t,new_rms,label='Corrected timing',color='#dc2626',alpha=.8)
    axes[1,col].plot(t,change,color='#334155')
    axes[0,col].set(title=f'{name}: Soft-HFC evoked response',ylabel='Spatial RMS (fT)')
    axes[1,col].set(title='Corrected minus historical waveform',ylabel='Spatial RMS of difference (fT)',xlabel='Time from event (s)')
    axes[0,col].legend()
    for row in [0,1]:axes[row,col].axvline(0,color='gray',lw=.6)
    report[name]=dict(relative_evoked_l2=float(np.linalg.norm(after-before)/np.linalg.norm(before)),
        times_sec=t.tolist(),old_spatial_rms=old_rms.tolist(),corrected_spatial_rms=new_rms.tolist(),difference_spatial_rms=change.tolist())
fig.savefig(root/'evoked_timing_effect.png',dpi=160)
write_report(root/'evoked_timing_effect.json',report)
print({k:v['relative_evoked_l2'] for k,v in report.items()})
