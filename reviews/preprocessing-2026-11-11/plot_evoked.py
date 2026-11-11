"""Inspect condition-average spatial RMS before/after HFC on replayed epochs."""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from opm_bridge.preprocessing.sensor_mapping import apply_soft_hfc
from opm_bridge.preprocessing.validation import write_report

out=Path('reviews/preprocessing-2026-11-11')
fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
metrics={}
for ax,(dataset,file) in zip(axs,[('FACE-DEC','data/cache/validation/ds005107_nohfc/sub-01/sub-01_ses-01_run-01_256hz.pt'),('Nottingham','data/cache/validation/nottingham/sub-101_ses-001_task-braille_run-001_256hz.pt')]):
    p=torch.load(file,weights_only=False)
    x=p['x' if dataset=='FACE-DEC' else 'x_raw_ft'].numpy()
    filtered=apply_soft_hfc(x,p['pos_ori'].numpy()[:,3:])
    if dataset=='FACE-DEC':
        labels=p['labels']['gender'].numpy(); unit='normalized units'; names=['gender label 0','gender label 1']
    else:
        labels=p['labels'].numpy();unit='fT';names=['index','pinky']
    t=-.2+np.arange(512)/256
    metrics[dataset]={}
    for label,color in zip([0,1],['#2563eb','#dc2626']):
        before=x[labels==label].mean(axis=0)
        after=filtered[labels==label].mean(axis=0)
        for values,style,suffix in [(before,'--','before HFC'),(after,'-','after HFC')]:
            rms=np.sqrt(np.mean(values**2,axis=0))
            ax.plot(t,rms,color=color,ls=style,label=f'{names[label]}, {suffix}',alpha=.8)
        metrics[dataset][str(label)]=dict(unit=unit,n_trials=int((labels==label).sum()),
            relative_evoked_l2_change=float(np.linalg.norm(after-before)/np.linalg.norm(before)),
            times_sec=t.tolist(),spatial_rms_before=np.sqrt(np.mean(before**2,axis=0)).tolist(),spatial_rms_after=np.sqrt(np.mean(after**2,axis=0)).tolist())
    ax.set(title=dataset,xlabel='Time from implemented event (s)',ylabel=f'Evoked spatial RMS ({unit})',xlim=(-.2,1.8))
    ax.axvline(0,color='gray',lw=.7);ax.legend(fontsize=7)
fig.savefig(out/'evoked_hfc_comparison.png',dpi=160)
write_report(out/'evoked_hfc_comparison.json',metrics)
