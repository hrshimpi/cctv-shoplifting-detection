"""Compare imported expanded A/B runs without selecting thresholds on test data."""
import json
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT / 'outputs/matplotlib'))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scripts.train_colab import score
from src.models.bigru_autoencoder import BiGRUAutoencoder
from src.experiments.artifacts import run_paths
from src.experiments.quality_dataset import sha256

IDENTITY = ['subset','pose_file','person_id','camera_id','video_id','start_frame','end_frame']
METRICS = ['precision','recall','f1','roc_auc','pr_auc','false_positive_rate']


def main():
    torch.set_num_threads(2)
    base=ROOT / 'outputs/logs/research_v1'
    diagnostic=json.loads((base / 'frame_diagnostic/report.json').read_text())
    out=base / 'expanded_ab_comparison'; out.mkdir(parents=True,exist_ok=True)
    rows, verification, hashes, transitions = [], [], {}, []
    reference=None; reference_report=None
    for family in ('a','b'):
        for seed in (0,1,2):
            name=f'expanded_{family}_seed{seed}'
            folder,log=run_paths(name)
            report=json.loads((log / 'evaluation.json').read_text())
            frame=pd.read_csv(log / 'window_predictions.csv',dtype={k:str for k in IDENTITY[:5]}).set_index(IDENTITY).sort_index()
            if reference is None:
                reference,reference_report=frame,report
            if not frame.index.equals(reference.index) or not np.array_equal(frame.label,reference.label):
                raise ValueError('A/B test window identities or labels differ')
            if report['config'] != reference_report['config'] or report['split'] != reference_report['split']:
                raise ValueError('A/B configuration or source-video split differs')
            hashes[name]=sha256(folder / 'model.pt')
            f=diagnostic['runs'][name]
            if f['checkpoint_sha256'] != hashes[name]:
                raise ValueError('Stale frame diagnostic')
            for scope,metrics in [('window',report['metrics']['combined']),('frame_scored',f['metrics']['combined'])]:
                rows.append(dict(run=name,family=family,seed=seed,scope=scope,
                    **{k:metrics[k] for k in METRICS},
                    all_annotated_anomaly_recall=metrics.get('all_annotated_anomaly_recall'),
                    score_coverage=metrics.get('score_coverage')))
            if family=='b':
                _,alog=run_paths(f'expanded_a_seed{seed}')
                a=pd.read_csv(alog / 'window_predictions.csv',dtype={k:str for k in IDENTITY[:5]}).set_index(IDENTITY).sort_index()
                cross=pd.crosstab(a.outcome,frame.outcome)
                for a_outcome in cross.index:
                    for b_outcome in cross.columns:
                        transitions.append(dict(seed=seed,a_outcome=a_outcome,b_outcome=b_outcome,windows=int(cross.loc[a_outcome,b_outcome])))
                # Spot-check that GPU weights reproduce scores in the local CPU environment.
                saved=torch.load(folder / 'model.pt',map_location='cpu',weights_only=False)
                cfg=saved['config']
                model=BiGRUAutoencoder(58,cfg['hidden_size'],cfg['num_layers'],cfg['dropout'])
                model.load_state_dict(saved['model_state_dict'])
                package=ROOT / 'outputs/colab_handoff/experiment_b/retail_training_b'
                for subset in ('test_realworld','test_staged','train'):
                    with np.load(package / f'data/{subset}.npz',allow_pickle=False) as arrays:
                        if subset=='train':
                            indices=arrays['validation_indices'][:32]
                            expected=np.load(folder / 'validation_errors.npy',allow_pickle=False)[:32]
                        else:
                            indices=np.arange(32)
                            with np.load(folder / f'{subset}_scores.npz',allow_pickle=False) as scores:
                                expected=scores['scores'][indices]
                        actual=score(model,arrays['features'][indices],arrays['masks'][indices],32,torch.device('cpu'))
                    passed=bool(np.allclose(actual,expected,rtol=1e-4,atol=1e-6))
                    verification.append(dict(run=name,subset=subset,windows=len(indices),passed=passed,
                        max_absolute_difference=float(np.max(np.abs(actual-expected)))))
                    if not passed:
                        raise ValueError('GPU/CPU score verification failed')
    table=pd.DataFrame(rows); table.to_csv(out / 'per_seed_metrics.csv',index=False)
    pd.DataFrame(transitions).to_csv(out / 'prediction_transitions.csv',index=False)
    summary=table.groupby(['scope','family'])[METRICS].agg(['mean','std'])
    summary.to_csv(out / 'mean_std.csv')
    means=table.groupby(['scope','family'])[METRICS].mean()
    a=means.loc[('window','a')]; b=means.loc[('window','b')]
    findings=[f"Window recall: A {a.recall:.2%}, B {b.recall:.2%}; F1: A {a.f1:.4f}, B {b.f1:.4f}.",
        f"Window ROC-AUC: A {a.roc_auc:.4f}, B {b.roc_auc:.4f}. The difference is {b.roc_auc-a.roc_auc:+.4f}; three seeds on one fixed split do not establish statistical significance.",
        'Adding explicit joint displacements did not improve mean recall or F1 at the fixed normal-validation p95 protocol.' if b.recall<=a.recall and b.f1<=a.f1 else 'Metric changes are mixed; inspect all seeds, coverage and false alarms before interpreting the feature change.',
        'Normal and anomalous reconstruction scores remain weakly separated. Do not infer usefulness from high precision/AP on the anomaly-rich window cohort.',
        'No threshold or checkpoint was selected by test metrics. B was an exploratory follow-up after A test inspection; independent confirmation is required.']
    for family in ('a','b'):
        group=table[(table.family==family)&(table.scope=='frame_scored')]
        findings.append(f"{family.upper()} frame recall on scored frames {group.recall.mean():.2%}, all-annotated-anomaly recall {group.all_annotated_anomaly_recall.mean():.2%}, ROC-AUC {group.roc_auc.mean():.4f}, coverage {group.score_coverage.mean():.2%}.")
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for row_index,scope in enumerate(['window','frame_scored']):
        for ax,metric in zip(axes[row_index],['recall','f1','roc_auc']):
            for seed in (0,1,2):
                points=table[(table.seed==seed)&(table.scope==scope)].set_index('family').loc[['a','b'],metric]
                ax.plot(['A: positions','B: + motion'],points,marker='o',label=f'Seed {seed}')
            ax.set(title=f'{scope}: {metric}',ylabel=metric)
    axes[0,0].legend(); fig.tight_layout()
    plot=out / 'paired_metrics.png'; fig.savefig(plot,dpi=150); plt.close(fig)
    next_step='Audit pose/label alignment and obtain independently reviewed person/time annotations on separate retail clips before further architecture or threshold tuning. Preserve A and B as measured baselines; no new training is started.'
    result=dict(checkpoint_sha256=hashes,windows_identical=True,splits_and_configs_identical=True,
        means=[dict(scope=scope,family=family,**row.to_dict()) for (scope,family),row in means.iterrows()],
        cpu_verification=verification,cpu_verified_windows=sum(v['windows'] for v in verification),
        findings=findings,next_step=next_step,plots=[str(plot.relative_to(ROOT))],
        limitations=['Same split and correlated windows/frames; seed variation is descriptive, not an independent test population.',
                     'Additional input/output weights and feature composition differ; raw losses cannot be directly compared.',
                     'Video-level anomaly masks are not person-specific theft annotations.'])
    (out / 'comparison.json').write_text(json.dumps(result,indent=2))
    (out / 'comparison.md').write_text('\n'.join(['# Expanded A/B comparison','']+['- '+x for x in findings]+['','## Next step','',next_step,'','## Mean and sample standard deviation','',summary.to_string()])+'\n')
    print('\n'.join(findings)); print('CPU verification:',result['cpu_verified_windows'],'windows passed')


if __name__=='__main__':
    main()
