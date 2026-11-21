"""Build cached experiment visuals and a notebook without retraining models."""
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/matplotlib'))
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nbformat


def main():
    out=ROOT/'outputs/mnnit_clip_classification'
    report=json.loads((out/'evaluation.json').read_text())
    inventory=pd.read_csv(out/'inventory.csv')
    predictions=pd.read_csv(out/'predictions.csv')
    primary=report['metrics'][report['primary_model']]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,split in zip(axes,['validation','test']):
        r=primary[split];matrix=np.array([[r['tn'],r['fp']],[r['fn'],r['tp']]])
        ax.imshow(matrix,cmap='Blues',vmin=0,vmax=max(matrix.max(),20))
        for y in range(2):
            for x in range(2):ax.text(x,y,str(matrix[y,x]),ha='center',va='center',fontsize=17,color='black')
        ax.set(xticks=[0,1],xticklabels=['Normal','Positive'],yticks=[0,1],yticklabels=['Normal','Positive'],
            xlabel='Predicted class',ylabel='Reference class',title=f'{split.capitalize()} - {r["clips"]} clips')
    fig.suptitle('Classification Confusion Matrices',fontsize=12)
    fig.tight_layout();fig.savefig(out/'confusion.png',dpi=160);plt.close(fig)
    errors=predictions[(predictions.model==report['primary_model']) & (predictions.split=='test') & (predictions.label!=predictions.prediction)]
    errors.to_csv(out/'test_errors.csv',index=False)
    with np.load(out/'sampled_rgb.npz',allow_pickle=False) as samples:
        rgb=samples['frames'];ids=samples['clip_ids'].tolist()
    sheet=np.full((int(np.ceil(len(errors)/3))*230,3*330,3),245,dtype=np.uint8)
    for position,row in enumerate(errors.itertuples()):
        x=position%3*330;y=position//3*230
        frame=cv2.cvtColor(rgb[ids.index(row.clip_id),8],cv2.COLOR_RGB2BGR)
        sheet[y+45:y+225,x+5:x+325]=cv2.resize(frame,(320,180))
        name=f'{row.clip_id} | {"False positive" if row.label==0 else "False negative"} | {row.positive_score:.3f}'
        cv2.putText(sheet,name,(x+5,y+18),cv2.FONT_HERSHEY_SIMPLEX,.43,(20,20,20),1,cv2.LINE_AA)
        reference='Normal' if row.label==0 else 'Shoplifting'
        predicted='Normal' if row.prediction==0 else 'Shoplifting'
        cv2.putText(sheet,f'Reference: {reference} / Predicted: {predicted}',(x+5,y+36),cv2.FONT_HERSHEY_SIMPLEX,.36,(20,20,20),1,cv2.LINE_AA)
    if len(errors):cv2.imwrite(str(out/'test_error_examples.jpg'),sheet)
    lines=['# MNNIT staged clip classification results','',
        'Primary model: frozen ImageNet MobileNetV3-small frame embeddings with mean, standard deviation and absolute consecutive changes; train-only scaler and logistic head; fixed threshold 0.5.','',
        'Data: 182 originals, 179 retained after excluding 3 exact duplicates; 18 near-duplicate pairs linked within split groups. Train 107, validation 36, test 36. Actor/session IDs are unavailable.','',
        '| Model | Split | Accuracy | Recall | F1 | ROC-AUC | FP | FN |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for name,sets in report['metrics'].items():
        for split in ['validation','test']:
            r=sets[split]
            lines.append(f'| {name} | {split} | {r["accuracy"]:.2%} | {r["recall"]:.2%} | {r["f1"]:.3f} | {r["roc_auc"]:.3f} | {r["fp"]} | {r["fn"]} |')
    lines+=['','The validation/test gap is material. These scores are not comparable as improvements over RetailS and do not establish real-store detection quality.','']+['- '+x for x in report['limitations']]
    (out/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    cells=[]
    def md(text):cells.append(nbformat.v4.new_markdown_cell(text))
    def code(text):cells.append(nbformat.v4.new_code_cell(text))
    md('# Supervised staged retail clip classification\n\nInspect the MNNIT version 1 data, frozen split, lightweight supervised baseline and errors. This notebook reuses verified cached results; it does not retrain the encoder or replace the RetailS pose baselines. Labels apply to entire staged laboratory clips, not acting people or exact theft times.')
    code('from pathlib import Path\nimport json, hashlib, sys\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport cv2\nROOT = Path.cwd()\nif ROOT.name == "notebooks": ROOT = ROOT.parent\nassert (ROOT / "configs/mnnit_clip_v1.json").is_file()\nif str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))\nOUT = ROOT / "outputs/mnnit_clip_classification"\ndef digest(path):\n    with path.open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()\nprint("Python:", sys.executable)\nprint("Repository:", ROOT)')
    md('## 1 Verify the original data\n\nThe archive has 90 normal and 92 positive publisher-labeled clips. Every source frame was decoded by the preparation script. Original FPS is 30; sizes include 640x480 and 1920x1080. Three exact duplicates are excluded.')
    code('data = json.loads((OUT / "data_report.json").read_text())\nassert digest(ROOT / "data/action_sources/mnnit_v1/Dataset.zip") == data["archive_sha256"]\nassert digest(OUT / "inventory.csv") == data["inventory_sha256"]\nassert digest(OUT / "sampled_rgb.npz") == data["rgb_cache_sha256"]\ndisplay(pd.DataFrame([{k:v for k,v in data.items() if k in ["videos","canonical_videos","exact_duplicates_excluded","near_duplicate_pairs","decoded_frames","decoded_all_frames"]}]))\ninventory = pd.read_csv(OUT / "inventory.csv")\ndisplay(inventory.groupby(["class_name", "included"]).size().rename("clips").reset_index())')
    md('## 2 Inspect the scene and labels\n\nA sampled midpoint from each original clip is available in the contact sheets. These are laboratory scenes with repeated clothing and backgrounds. Preview inspection does not independently verify every action label.')
    code('for name in ["contact_01.jpg", "contact_06.jpg"]:\n    image = cv2.imread(str(OUT / name))\n    assert image is not None\n    plt.figure(figsize=(13, 12)); plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)); plt.axis("off"); plt.show()')
    md('## 3 Check the frozen split\n\nSeed 2026. Exact and detected sampled near-duplicate components remain within one split. The release lacks verified actor/session IDs; this does not demonstrate independence across people or recording sessions.')
    code('from src.data.clip_classification import verify_split, portable_probability\nsplit = json.loads((OUT / "frozen_split.json").read_text())\nverify_split(split["rows"])\ndisplay(pd.DataFrame(split["counts"]).T.rename(columns={"0":"normal", "1":"positive"}))\nconfig = json.loads((ROOT / "configs/mnnit_clip_v1.json").read_text())\nassert digest(ROOT / "configs/mnnit_clip_v1.json") == split["config_sha256"]\ndisplay(config)')
    md('## 4 Verify models and measured metrics\n\nThe primary model uses mean, standard deviation and absolute consecutive changes of 576-dimensional frozen frame embeddings. The control omits changes. The scaler and logistic head use training clips only; C=1, balanced classes and fixed threshold 0.5. No test-based tuning is performed.')
    code('report = json.loads((OUT / "evaluation.json").read_text())\nassert report["complete"] and not report["test_selected_parameters"] and not report["old_pose_models_changed"]\nassert digest(OUT / "frozen_split.json") == report["frozen_split_sha256"]\nassert digest(OUT / "predictions.csv") == report["predictions_sha256"]\nfor name, expected in report["model_sha256"].items():\n    assert digest(OUT / (name + "_model.npz")) == expected\nrows = [{"model":name,"split":subset,**values} for name, subsets in report["metrics"].items() for subset, values in subsets.items()]\ndisplay(pd.DataFrame(rows)[["model","split","clips","accuracy","balanced_accuracy","precision","recall","f1","roc_auc","tp","tn","fp","fn"]])')
    md('## 5 Examine confusion counts and errors\n\nThere are 36 test clips. The primary model has 7 false positives and 4 false negatives. Validation accuracy is higher than test accuracy; these results remain provisional. Test-error examples are diagnostic, not a new tuning set.')
    code('for name in ["confusion.png", "test_error_examples.jpg"]:\n    image = cv2.imread(str(OUT / name))\n    assert image is not None\n    plt.figure(figsize=(13, 10)); plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)); plt.axis("off"); plt.show()\ndisplay(pd.read_csv(OUT / "test_errors.csv"))')
    md('## 6 Check portable inference\n\nThe NPZ contains the scaler and logistic coefficients. Scores should reproduce the saved predictions without loading a pickle or training a model. Scores are not calibrated theft probabilities.')
    code('with np.load(OUT / "frozen_features.npz", allow_pickle=False) as features:\n    matrix = features["features"]; clip_ids = features["clip_ids"].tolist()\nwith np.load(OUT / "appearance_and_changes_model.npz", allow_pickle=False) as model:\n    probability = portable_probability(matrix, model["mean"], model["scale"], model["coefficient"], model["intercept"])\npredictions = pd.read_csv(OUT / "predictions.csv")\nprimary = predictions[predictions.model == report["primary_model"]]\nactual = np.array([probability[clip_ids.index(clip)] for clip in primary.clip_id])\nnp.testing.assert_allclose(actual, primary.positive_score, rtol=1e-7, atol=1e-9)\nprint("Portable inference matches all", len(primary), "retained clip scores.")\nprint("Full regression evidence:", OUT / "regression_tests.txt")')
    md('## Interpretation and next research gate\n\nThis completes a lightweight supervised feasibility experiment. The dataset, supervision and split differ from RetailS, so its metrics are not proof of improvement on the prior benchmark. Actor/session independence, real-store generalization and person/time localization remain unverified. Temporal pooling loses detailed order, and feature changes are not optical flow. Preserve the exposed test split as historical evidence. A stronger temporal model should be evaluated under a fresh, predeclared independent protocol.\n\nSource and attribution: Ansari and Singh, Mendeley Data V1, DOI 10.17632/r3yjf35hzr.1, CC BY 4.0. See MNNIT_CLIP_CLASSIFICATION.md for reproduction and runtime details.')
    notebook=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3.11 (Shoplifting)','language':'python','name':'shoplifting-py311'},'language_info':{'name':'python','version':'3.11'}})
    target=ROOT/'notebooks/Retail_Shoplifting_Supervised_Clips.ipynb'
    nbformat.write(notebook,target)
    print('Built visuals and notebook:',target)


if __name__=='__main__':main()
