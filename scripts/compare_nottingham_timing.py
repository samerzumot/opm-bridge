#!/usr/bin/env python3
"""Paired sub-101 timing sensitivity, preserving historical benchmark results.

Uses the historical MAG/Soft-HFC monolithic and axis-partitioned protocols,
plus the classical decoder with and without Soft-HFC. Same CPU model, trial
order and five-fold splits are used for both timing conventions.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from opm_bridge.eval.metrics import eval_linear_probe_from_embeddings
from opm_bridge.models.baselines import ClassicalBandPowerDecoder
from opm_bridge.models.wrapper import load_brainomni_backbone, BRAINOMNI_HF_REPO, BRAINOMNI_HF_REVISION
from opm_bridge.preprocessing.sensor_mapping import apply_soft_hfc, sensortype_wise_normalize
from opm_bridge.preprocessing.validation import write_report
from scripts.replay_hfc_validation import difference
from scripts.run_nottingham_tag_partition_factorial import reconstruct_and_encode_partition, compute_per_trial_pcc


def decode(features, labels, splits):
    scores=[]
    for train,test in splits:
        metric=eval_linear_probe_from_embeddings(features[train],labels[train],features[test],labels[test],seed=42)
        scores.append(float(metric['balanced_accuracy']))
    return dict(fold_balanced_accuracy=scores,mean_balanced_accuracy=float(np.mean(scores)))


def evaluate(model, x, pos, groups, labels, splits):
    scores=[]; embeddings=[]
    with torch.inference_mode():
        for start in range(0,len(x),16):
            xb=torch.from_numpy(x[start:start+16]).float()
            pb=torch.from_numpy(pos).float()[None].expand(len(xb),-1,-1)
            sb=torch.ones((len(xb),x.shape[1]),dtype=torch.long)
            target=model.tokenizer.norm_target(model.tokenizer.unfold(xb,overlap_ratio=0.))
            rec=torch.zeros_like(target); latent=[]
            for group in groups:
                reconstructed,h=reconstruct_and_encode_partition(model,xb[:,group],pb[:,group],sb[:,group],torch.device('cpu'))
                rec[:,group]=reconstructed
                latent.append(h)
            h=torch.cat(latent,dim=1)
            embeddings.append(h[:,:,1:3,:].mean(dim=2).reshape(len(xb),-1).numpy())
            scores.extend(compute_per_trial_pcc(rec,target).tolist())
            print(f'  trials {min(start+16,len(x))}/{len(x)}',flush=True)
    return dict(trial_pccs=scores,pcc_trial_weighted=float(np.mean(scores)),
                pretrained_linear_probe=decode(np.concatenate(embeddings),labels,splits))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old',required=True);parser.add_argument('--corrected',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42)
    payloads={key:torch.load(path,map_location='cpu',weights_only=False) for key,path in [('old',args.old),('corrected',args.corrected)]}
    qc={key:json.loads(Path(path).with_suffix('.qc.json').read_text()) for key,path in [('old',args.old),('corrected',args.corrected)]}
    old,new=payloads['old'],payloads['corrected']
    for key in ['labels','pos_ori','sensor_type']:
        if not torch.equal(old[key],new[key]):raise ValueError(f'Unexpected {key} change')
    if qc['old']['provenance']['sources']!=qc['corrected']['provenance']['sources']:
        raise ValueError('Source bytes differ')
    for key in ['settings','filter','versions','sfreq_raw','baseline_sec','soft_hfc_alpha','order']:
        if qc['old']['provenance'][key]!=qc['corrected']['provenance'][key]:
            raise ValueError(f'Unexpected preprocessing difference: {key}')
    source_events=qc['old']['retained_events']; corrected_events=qc['corrected']['retained_events']
    if len(source_events)!=len(corrected_events) or any(a['sample']-b['sample']!=1 or a['label']!=b['label'] for a,b in zip(source_events,corrected_events)):
        raise ValueError('Expected every event to move exactly one sample earlier, preserving labels')
    x_old=old['x_raw_ft'].numpy();x_new=new['x_raw_ft'].numpy()
    pos=new['pos_ori'].numpy();labels=new['labels'].numpy();tags=new['sensor_type'].numpy()
    if x_old.shape!=x_new.shape:raise ValueError('Epoch shape changed')
    splits=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(x_new,labels))
    report=dict(subject=new['subject_id'],scope='Representative sub-101; no cohort inference',
        checkpoint=dict(repo=BRAINOMNI_HF_REPO,revision=BRAINOMNI_HF_REVISION),device='cpu',
        versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','mne','torch','scikit-learn']},
        caches={k:dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for k,p in [('old',args.old),('corrected',args.corrected)]},
        event_shift_native_samples=-1,event_shift_seconds=-1/qc['corrected']['provenance']['sfreq_raw'],
        shape=list(x_new.shape),class_counts=np.bincount(labels).tolist(),source_hashes_match=True,
        epoch_difference=difference(x_old,x_new),fold_test_indices=[test.tolist() for _,test in splits],
        classical={},reconstruction={},historical_comparisons={})
    write_report(args.out,report)
    normalized={}
    dec=ClassicalBandPowerDecoder(sfreq=256.)
    for variant,payload in payloads.items():
        x=payload['x_raw_ft'].numpy()
        for filtered in ['raw','softhfc']:
            values=apply_soft_hfc(x,pos[:,3:],.25) if filtered=='softhfc' else x.copy()
            values=np.stack([sensortype_wise_normalize(v,tags) for v in values])
            report['classical'][f'{variant}_{filtered}']=decode(dec.extract_features(values[:,:,64:268]),labels,splits)
            if filtered=='softhfc':normalized[variant]=values
    report['normalized_input_difference']=difference(normalized['old'],normalized['corrected'])
    write_report(args.out,report)
    model=load_brainomni_backbone().cpu().eval()
    axes=new['meta'].axis_labels
    groups={'monolithic':[list(range(x_new.shape[1]))],
            'axis_partitioned':[[i for i,a in enumerate(axes) if a==axis] for axis in ('Z','X','Y')]}
    historical=json.loads(Path('results/summary_phase4_nottingham_factorial_and_seeds.json').read_text())['per_subject'][new['subject_id']]['softhfc']['mag']
    for mode,partition in groups.items():
        report['reconstruction'][mode]={}
        for variant in ['old','corrected']:
            print(mode,variant,flush=True)
            report['reconstruction'][mode][variant]=evaluate(model,normalized[variant],pos,partition,labels,splits)
            write_report(args.out,report)
        pair=report['reconstruction'][mode];reference=historical[mode]
        pair['delta_pcc']=pair['corrected']['pcc_trial_weighted']-pair['old']['pcc_trial_weighted']
        pair['delta_probe_bacc']=pair['corrected']['pretrained_linear_probe']['mean_balanced_accuracy']-pair['old']['pretrained_linear_probe']['mean_balanced_accuracy']
        report['historical_comparisons'][mode]=dict(
            reported_pcc=reference['pcc_trial_weighted'],
            old_replay_minus_reported_pcc=pair['old']['pcc_trial_weighted']-reference['pcc_trial_weighted'],
            corrected_minus_reported_pcc=pair['corrected']['pcc_trial_weighted']-reference['pcc_trial_weighted'],
            reported_probe_bacc=reference['braille_5fold_bacc']['pretrained_linear_probe'],
            old_replay_probe_bacc=pair['old']['pretrained_linear_probe']['mean_balanced_accuracy'],
            corrected_probe_bacc=pair['corrected']['pretrained_linear_probe']['mean_balanced_accuracy'])
        write_report(args.out,report)
    for variant in ['old','corrected']:
        report[f'{variant}_axis_minus_monolithic']=report['reconstruction']['axis_partitioned'][variant]['pcc_trial_weighted']-report['reconstruction']['monolithic'][variant]['pcc_trial_weighted']
    report['delta_axis_advantage']=report['corrected_axis_minus_monolithic']-report['old_axis_minus_monolithic']
    write_report(args.out,report)


if __name__=='__main__':main()
