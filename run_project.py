"""Verify saved models, rebuild the results page and open it locally.

From the VS Code terminal in this repository:
    .\.venv\Scripts\python.exe run_project.py

Use --check to verify models and regenerate the page without starting a server.
This launcher uses existing artifacts and does not retrain models or download data.
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def verify_and_predict():
    import numpy as np
    from src.data.clip_classification import portable_probability,verify_split
    folder=ROOT/'outputs/mnnit_clip_classification'
    report=json.loads((folder/'evaluation.json').read_text())
    if not report['complete']:raise ValueError('Supervised experiment is incomplete')
    split_path=folder/'frozen_split.json'
    if digest(split_path)!=report['frozen_split_sha256']:raise ValueError('Saved split was modified')
    split=json.loads(split_path.read_text());verify_split(split['rows'])
    if digest(ROOT/'configs/mnnit_clip_v1.json')!=split['config_sha256']:
        raise ValueError('Configuration differs from the saved experiment')
    for name,expected in report['model_sha256'].items():
        if digest(folder/(name+'_model.npz'))!=expected:raise ValueError('Saved model was modified')
    if digest(folder/'predictions.csv')!=report['predictions_sha256']:
        raise ValueError('Saved prediction evidence was modified')
    data=json.loads((folder/'data_report.json').read_text())
    if digest(folder/'inventory.csv')!=data['inventory_sha256']:
        raise ValueError('Dataset inventory was modified')
    with np.load(folder/'frozen_features.npz',allow_pickle=False) as features:
        values=features['features'];ids=features['clip_ids'].tolist()
    with np.load(folder/(report['primary_model']+'_model.npz'),allow_pickle=False) as head:
        scores=portable_probability(values,head['mean'],head['scale'],head['coefficient'],head['intercept'])
    with (folder/'predictions.csv').open(newline='',encoding='utf-8') as stream:
        expected=[r for r in csv.DictReader(stream) if r['model']==report['primary_model']]
    actual=np.array([scores[ids.index(r['clip_id'])] for r in expected])
    np.testing.assert_allclose(actual,np.array([float(r['positive_score']) for r in expected]),rtol=1e-7,atol=1e-9)
    test=[r for r in expected if r['split']=='test']
    correct=sum((float(scores[ids.index(r['clip_id'])])>=report['threshold'])==bool(int(r['label'])) for r in test)
    measured=report['metrics'][report['primary_model']]['test']
    if not np.isclose(correct/len(test),measured['accuracy']):raise ValueError('Fresh inference disagrees with saved accuracy')
    print(f'PASS: saved model inference matches all {len(expected)} retained clips.')
    print(f"Test: {correct}/{len(test)} correct | accuracy {measured['accuracy']:.2%} | recall {measured['recall']:.2%} | F1 {measured['f1']:.3f} | ROC-AUC {measured['roc_auc']:.3f}")
    print('Data: staged laboratory clip classification. Existing pose models remain unchanged.')
    return report


def server_identity(url,manifest_id):
    try:
        with urllib.request.urlopen(url+'/review_manifest.json',timeout=2) as response:
            return json.load(response).get('manifest_id')==manifest_id
    except (OSError,ValueError,urllib.error.URLError):return False


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check',action='store_true',help='Check saved inference and build reports, without starting a server')
    parser.add_argument('--no-browser',action='store_true',help='Start or reuse the server without opening a browser tab')
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    if not 1024<=args.port<=65535:parser.error('--port must be between 1024 and 65535')
    print('Repository:',ROOT,flush=True)
    print('Using cached models and data; no training or downloads.',flush=True)
    verify_and_predict()
    folder=ROOT/'outputs/mnnit_clip_classification'
    if not (folder/'confusion.png').exists() or not (folder/'test_error_examples.jpg').exists():
        subprocess.run([sys.executable,str(ROOT/'scripts/build_mnnit_clip_report.py')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/build_autonomous_review_report.py')],cwd=ROOT,check=True)
    if args.check:
        print('PASS: verified inference and rebuilt the results page.');return
    base=f'http://127.0.0.1:{args.port}'
    manifest=json.loads((ROOT/'outputs/manual_validation/review_manifest.json').read_text())
    if not server_identity(base,manifest['manifest_id']):
        stdout=(folder/'local_server_stdout.log').open('a',encoding='utf-8')
        stderr=(folder/'local_server_stderr.log').open('a',encoding='utf-8')
        try:
            server=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/serve_validation_review.py'),'--port',str(args.port)],
                cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
        finally:stdout.close();stderr.close()
        for _ in range(50):
            if server_identity(base,manifest['manifest_id']):break
            if server.poll() is not None:
                raise RuntimeError('Results server could not start. Check outputs/mnnit_clip_classification/local_server_stderr.log or choose another --port.')
            time.sleep(.2)
        else:raise RuntimeError('Timed out waiting for the local results server')
        (folder/'local_server.json').write_text(json.dumps({'pid':server.pid,'port':args.port},indent=2))
    url=base+'/index.html?view=results&phase=local-run#overview'
    with urllib.request.urlopen(base+'/media/mnnit_evaluation.json',timeout=5) as response:
        served=json.load(response)
    if served['predictions_sha256']!=json.loads((folder/'evaluation.json').read_text())['predictions_sha256']:
        raise ValueError('Browser server is serving a different experiment')
    print('Results:',url)
    if not args.no_browser:webbrowser.open(url,new=2)
    print('Ready. The local server continues running in the background.')


if __name__=='__main__':
    try:main()
    except (FileNotFoundError,ImportError) as error:
        raise SystemExit('Required local project artifact or dependency is missing. Use this repository and its .venv. '+str(error))
