"""Export stage 05 measured curves and tables; performs no training."""
import csv
import json
from pathlib import Path
import statistics
from collections import Counter
import math

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'artifacts/05_pilot'
LOG = ROOT/'reports/05_logs'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[json.loads(line) for line in (ART/'ce_run/metrics.jsonl').read_text().splitlines()]
    train=[r for r in rows if r['event']=='train']
    val=[r for r in rows if r['event']=='validation']
    assert len(train)==300 and train[-1]['step']==300
    fields=['step','loss','ce','embedding_mse','residue_accuracy','sequence_accuracy','residue_count','sequence_count']
    for name,data in [('training_metrics',train),('validation_metrics',val)]:
        with (LOG/f'{name}.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
            writer.writeheader();writer.writerows(data)
    with (LOG/'validation_by_length.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['step','length_bucket',*fields[1:]],extrasaction='ignore')
        writer.writeheader()
        for row in val:
            for key,values in row['by_length'].items():
                if values is not None:
                    writer.writerow({'step':row['step'],'length_bucket':key,**values})
    def weighted_window(window):
        residues=sum(r['residue_count'] for r in window)
        sequences=sum(r['sequence_count'] for r in window)
        return {'steps':[window[0]['step'],window[-1]['step']],
                **{key:sum(r[key]*r['residue_count'] for r in window)/residues for key in ('ce','embedding_mse','residue_accuracy')},
                'sequence_accuracy':sum(r['sequence_accuracy']*r['sequence_count'] for r in window)/sequences}
    summary={'initial_validation':val[0],'final_validation':val[-1],
             'best_validation_ce':min(val,key=lambda r:r['ce']),
             'training_first_25':weighted_window(train[:25]),'training_last_25':weighted_window(train[-25:]),
             'optimizer_step_seconds_median':statistics.median(r['optimizer_step_seconds_including_cache_io'] for r in train),
             'note':'Training values precede each update on varying shuffled batches. Validation follows the indicated update. Only step 300 checkpoint is retained.'}
    (LOG/'curve_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    from templatedf.data import load_fasta,AA_ORDER
    train_records=load_fasta(ART/'train.fasta')[0]
    val_records=load_fasta(ART/'validation.fasta')[0]
    counts=Counter(''.join(r.sequence for r in train_records))
    total=sum(counts.values())
    probs={aa:(counts[aa]+1)/(total+20) for aa in AA_ORDER}
    majority=max(AA_ORDER,key=lambda aa:counts[aa])
    val_count=sum(r.length for r in val_records)
    baseline={'method':'Position-independent training AA frequency; add-one smoothing; no optimization',
              'training_residue_count':total,'validation_residue_count':val_count,'training_aa_counts':dict(counts),
              'majority_residue':majority,
              'validation_ce':sum(-math.log(probs[aa]) for r in val_records for aa in r.sequence)/val_count,
              'validation_residue_accuracy':sum(r.sequence.count(majority) for r in val_records)/val_count,
              'validation_sequence_accuracy':sum(r.sequence.count(majority)/r.length for r in val_records)/len(val_records)}
    (LOG/'composition_baseline.json').write_text(json.dumps(baseline,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    tx=[r['step'] for r in train];vx=[r['step'] for r in val]
    for ax,key,title in [(axes[0,0],'ce','Cross-entropy'),(axes[0,1],'residue_accuracy','Residue accuracy'),(axes[1,0],'embedding_mse','Embedding MSE (not supervised; lambda=0)')]:
        ax.plot(tx,[r[key] for r in train],alpha=.4,linewidth=.8,label='Train batch (before update)')
        ax.plot(vx,[r[key] for r in val],marker='o',linewidth=1.5,label='Validation (128 peptides)')
        ax.set(title=title,xlabel='Optimizer step');ax.grid(alpha=.2);ax.legend(fontsize=8)
    for ax,key in [(axes[0,0],'validation_ce'),(axes[0,1],'validation_residue_accuracy')]:
        ax.axhline(baseline[key],color='black',linestyle='--',linewidth=.8,label='Train-composition baseline')
        ax.legend(fontsize=8)
    axes[0,1].set_ylim(0,1)
    for key in val[0]['by_length']:
        axes[1,1].plot(vx,[r['by_length'][key]['residue_accuracy'] for r in val],marker='.',label=f'{key} aa')
    axes[1,1].set(title='Validation residue accuracy by length',xlabel='Optimizer step',ylim=(0,1))
    axes[1,1].grid(alpha=.2);axes[1,1].legend(fontsize=8)
    fig.suptitle('Real peptide 1A pilot | 512 train / 128 validation | D1280 N50 4+4 | BF16 batch 8\nBounded 300-step pilot; not a converged or high-quality generation model',fontsize=12)
    fig.savefig(LOG/'training_curves.png',dpi=160)
    fig.savefig(LOG/'training_curves.svg')
    plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!='best_validation_ce'},indent=2))


if __name__=='__main__':
    main()
