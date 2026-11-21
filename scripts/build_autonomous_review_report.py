"""Build the research results dashboard from verified experiment artifacts."""
import base64
import html
import hashlib
import json
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.readable_video_ui import load_report,overview,card as readable_card,CSS,JAVASCRIPT


def main():
    folder=ROOT/'outputs/manual_validation'
    review=json.loads((folder/'ai_review.json').read_text())
    manifest=json.loads((folder/'review_manifest.json').read_text())
    analysis=json.loads((ROOT/'outputs/retail_pilot_analysis/report.json').read_text())
    videos=json.loads((ROOT/'outputs/video_analysis/verification.json').read_text())
    video_by_id={v['case_id']:v for v in videos}
    if set(video_by_id)!={c['case_id'] for c in manifest['cases']}:
        raise ValueError('Annotated video set is incomplete')
    if review['manifest_id']!=analysis['manifest_id'] or manifest['manifest_id']!=review['manifest_id']:
        raise ValueError('Review and analysis identities differ')
    def escape(value):return html.escape(str(value))
    supervised_section=''
    overview_metrics=''
    supervised=ROOT/'outputs/mnnit_clip_classification'
    if (supervised/'evaluation.json').exists():
        result=json.loads((supervised/'evaluation.json').read_text())
        if not result['complete'] or result['test_selected_parameters'] or result['old_pose_models_changed']:
            raise ValueError('Supervised baseline is incomplete or changed the frozen protocol')
        for name,digest in result['model_sha256'].items():
            if hashlib.sha256((supervised/(name+'_model.npz')).read_bytes()).hexdigest()!=digest:
                raise ValueError('Supervised head artifact changed')
        if hashlib.sha256((supervised/'frozen_split.json').read_bytes()).hexdigest()!=result['frozen_split_sha256']:
            raise ValueError('Supervised split changed')
        primary=result['metrics'][result['primary_model']]
        score_rows=''.join(f"<tr><th scope='row'>{subset.capitalize()}</th><td>{r['clips']}</td><td>{r['accuracy']:.2%}</td><td>{r['recall']:.2%}</td><td>{r['f1']:.3f}</td><td>{r['roc_auc']:.3f}</td><td>{r['fp']}</td><td>{r['fn']}</td></tr>" for subset,r in primary.items())
        test=primary['test']
        overview_metrics=''.join(f'<div class="metric"><span>{label}</span><strong>{value}</strong><small>Held-out MNNIT clips</small></div>' for label,value in [('Test accuracy',f"{test['accuracy']:.2%}"),('Recall',f"{test['recall']:.2%}"),('F1 score',f"{test['f1']:.3f}"),('ROC-AUC',f"{test['roc_auc']:.3f}")])
        confusion_uri='data:image/png;base64,'+base64.b64encode((supervised/'confusion.png').read_bytes()).decode()
        error_uri='data:image/jpeg;base64,'+base64.b64encode((supervised/'test_error_examples.jpg').read_bytes()).decode()
        exports=[]
        for name in ['evaluation.json','data_report.json','predictions.csv','test_errors.csv','results.md']:
            target=folder/'media'/('mnnit_'+name);shutil.copy2(supervised/name,target)
            labels={'evaluation.json':'Evaluation summary (JSON)','data_report.json':'Data verification (JSON)','predictions.csv':'Predictions (CSV)','test_errors.csv':'Test errors (CSV)','results.md':'Results summary (Markdown)'}
            exports.append(f'<a download href="media/{escape(target.name)}">{labels[name]}</a>')
        word=ROOT/'outputs/documents/Shoplifting_Experiment_Progress_One_Page.docx'
        if word.exists():
            shutil.copy2(word,folder/'media'/'experiment_progress.docx')
            exports.insert(0,'<a download href="media/experiment_progress.docx">Experiment report (Word)</a>')
        supervised_section=f'''<section id="supervised-clips" aria-labelledby="classification-title"><div class="section-heading"><span class="eyebrow">Experiment 02</span><h2 id="classification-title">Supervised clip classification</h2></div>
<p>The MNNIT dataset contains 182 staged laboratory videos: 90 normal and 92 shoplifting-labeled clips. Three exact duplicates were removed, leaving 179 clips. Detected near-duplicates were grouped within the same split.</p>
<div class="tags"><span>107 training clips</span><span>36 validation clips</span><span>36 test clips</span></div>
<p>Pretrained MobileNetV3-small image features and temporal feature changes are classified using logistic regression. The encoder is fixed, preprocessing is fitted on training data, and the classification threshold is 0.5.</p>
<div class="scroll"><table><caption>Classification results by data split</caption><thead><tr><th scope="col">Split</th><th scope="col">Clips</th><th scope="col">Accuracy</th><th scope="col">Recall</th><th scope="col">F1</th><th scope="col">ROC-AUC</th><th scope="col">FP</th><th scope="col">FN</th></tr></thead><tbody>{score_rows}</tbody></table></div>
<p class="context">The test set contains 18 normal and 18 positive clips: 25 correct classifications, seven false positives and four false negatives. Validation accuracy is 94.44%, compared with 69.44% on the test set. These clip-level results use a different dataset and evaluation task from the RetailS pose experiments.</p>
<figure><img src="{confusion_uri}" alt="Validation and test confusion matrices for MNNIT clip classification"><figcaption>Figure 1. Reference and predicted class counts for the validation and test splits.</figcaption></figure>
<details><summary>Test error examples</summary><p>All 11 test errors are shown below using one sampled frame per clip. Labels refer to the full source clip; classifier scores are uncalibrated.</p><img src="{error_uri}" alt="Examples of all seven false positives and four false negatives"></details>
<details><summary>Dataset verification and evaluation protocol</summary><p>All 60,702 source frames decoded successfully, and the archive matches the publisher SHA256. Eighteen sampled near-duplicate pairs were linked within split groups. Actor and recording-session independence cannot be established from the available metadata. Repeated backgrounds and clothing may influence the results.</p><p>Clip descriptors contain the mean, standard deviation and absolute consecutive changes of image embeddings. These are feature changes, not optical flow or physical velocity. No test-based parameter or threshold search was performed.</p></details>
<div class="downloads"><h3>Report downloads</h3>{' '.join(exports)}</div>
<p class="source">Dataset: Ansari and Singh, <a href="https://data.mendeley.com/datasets/r3yjf35hzr/1">Mendeley Data V1, DOI 10.17632/r3yjf35hzr.1</a>. License: CC BY 4.0.</p></section>'''
    failure_section=''
    failure_folder=ROOT/'outputs/model_failure_audit'
    if (failure_folder/'report.json').exists():
        failure=json.loads((failure_folder/'report.json').read_text())
        if not failure['complete'] or failure['changes_to_models_or_thresholds']:
            raise ValueError('Failure audit is incomplete or changed frozen models')
        for name,record in failure['models'].items():
            if record['checkpoint_sha256']!=analysis['checkpoint_sha256'][name]:
                raise ValueError('Failure audit and video checkpoints differ')
        plot_uri='data:image/png;base64,'+base64.b64encode((failure_folder/'failure_summary.png').read_bytes()).decode()
        downloads=[]
        for name in ['report.json','window_label_support_and_predictions.csv','mask_interval_coverage.csv','frozen_run_errors.csv','diagnostic_queue.csv','alerts_by_positive_frame_count.csv']:
            target=folder/'media'/('failure_audit_'+name)
            shutil.copy2(failure_folder/name,target)
            downloads.append(f'<a download href="media/{escape(target.name)}">{escape(name)}</a>')
        items=''.join('<li>'+escape(f)+'</li>' for f in failure['findings'])
        failure_section=f'''<section id="failure-audit"><span class="eyebrow">Error analysis</span><h2>Pose model errors and label support</h2>
<p>Across the six expanded A/B models, 1,476 of 1,754 positive windows were missed by every model, while 13 normal windows triggered all six models. Mixed frame labels and incomplete pose coverage were examined separately.</p>
<figure><img src="{plot_uri}" alt="Pose model recall and positive windows missed by all six models"><figcaption>Figure 2. Retained-window results from the frozen RetailS pose models.</figcaption></figure>
<details><summary>Detailed findings and supporting data</summary><ul>{items}</ul><div class="downloads">{' '.join(downloads)}</div></details></section>'''
    readable=load_report(ROOT,manifest)
    activity_overview=overview(readable) if readable else ''
    cards=[]
    for case,note in zip(manifest['cases'],review['reviews']):
        video=video_by_id[case['case_id']]
        video_path=ROOT/video['video']
        if hashlib.sha256(video_path.read_bytes()).hexdigest()!=video['video_sha256'] or video['decoded_output_frames']!=case['frame_count']:
            raise ValueError('Annotated video verification failed')
        video_url=video_path.relative_to(folder).as_posix()
        poster_url=(ROOT/video['poster']).relative_to(folder).as_posix()
        rows=[]
        for r in analysis['comparisons']:
            if r['case_id']==case['case_id'] and r['run'].endswith('seed0'):
                rows.append(f"<tr><td>{escape(r['run'])}</td><td>{escape(r['mode'])}</td><td>{r['accepted_windows']}</td><td>{r['score_coverage']:.1%}</td><td>{r['flagged_windows']}</td></tr>")
        if readable:
            cards.append(readable_card(ROOT,case,note,readable[case['case_id']],video_url,''.join(rows)))
            continue
        cards.append(f'''<article class="video-card" id="{escape(case['case_id'].lower())}"><div class="video-heading"><h3>Video {escape(case['case_id'])}</h3><span class="badge">Tracking: {escape(note['tracking_quality'].replace('_',' ').capitalize())}</span></div>
<video controls preload="metadata" poster="{escape(poster_url)}" src="{escape(video_url)}"></video>
<p class="video-meta"><span>{video['decoded_output_frames']:,} decoded frames</span><span>{video['output_fps']:g} FPS</span><span>{video['unverified_alert_frames']} model-alert frames</span></p>
<details><summary>Scene observations and source video</summary><p>{escape(note['notes'])}</p><video controls preload="none" src="{escape(case['source_file'])}"></video>
<div class="scroll"><table><thead><tr><th>Model</th><th>Processing</th><th>Accepted windows</th><th>Score coverage</th><th>Flagged windows</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></details></article>''')
    image=ROOT/analysis['plots'][0]
    image_uri='data:image/png;base64,'+base64.b64encode(image.read_bytes()).decode()
    video_links=''.join(f'<a href="#{escape(c["case_id"].lower())}">{escape(c["case_id"])}</a>' for c in manifest['cases'])
    page=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>Retail Shoplifting Detection | Experimental Results</title><style>
:root{{color-scheme:light;--ink:#172637;--muted:#526476;--line:#dce4ec;--accent:#24577e}}*{{box-sizing:border-box}}html{{scroll-padding-top:20px}}body{{margin:0;background:#f3f6fa;color:var(--ink);font:16px/1.65 'Segoe UI',Arial,sans-serif}}main{{max-width:1220px;margin:auto;padding:42px 28px 24px}}header{{padding:10px 0 24px}}.eyebrow{{display:block;color:var(--muted);font-size:12px;letter-spacing:.12em;text-transform:uppercase;font-weight:650}}h1{{font-size:clamp(28px,4vw,42px);line-height:1.2;font-weight:680;letter-spacing:-.025em;margin:10px 0}}.subtitle{{margin:0;color:var(--muted);font-size:18px}}nav{{display:flex;flex-wrap:wrap;gap:10px 24px;border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:14px 0;margin:0 0 24px}}nav a{{color:var(--accent);font-size:14px;font-weight:600;text-decoration:none}}a{{color:var(--accent)}}a:hover{{text-decoration:underline}}a:focus-visible,summary:focus-visible{{outline:3px solid #83aac8;outline-offset:4px}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:8px 0 28px}}.metric{{background:white;border:1px solid var(--line);border-radius:10px;padding:17px 20px}}.metric span,.metric small{{display:block;color:var(--muted);font-size:13px}}.metric strong{{display:block;font-size:32px;line-height:1.4;letter-spacing:-.025em}}.metric small{{font-size:11px}}section{{background:white;padding:28px;border:1px solid var(--line);border-radius:12px;margin:22px 0}}h2{{font-size:24px;line-height:1.3;letter-spacing:-.015em;margin:6px 0 16px}}h3{{font-size:17px;margin:0 0 12px}}p{{margin:12px 0}}.tags,.video-meta{{display:flex;gap:10px;flex-wrap:wrap}}.tags span,.badge{{background:#edf3f8;border:1px solid #dde7ef;border-radius:5px;padding:4px 10px;font-size:12px;color:#354d63}}.tags{{margin:18px 0}}table{{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}}caption{{text-align:left;color:var(--muted);font-size:13px;padding:8px 0 10px}}td,th{{text-align:right;border-bottom:1px solid var(--line);padding:11px 13px;white-space:nowrap}}td:first-child,th:first-child{{text-align:left}}thead th{{background:#edf3f8;font-weight:650}}tbody th{{font-weight:600}}tbody tr:last-child td,tbody tr:last-child th{{border-bottom:0}}.scroll{{overflow-x:auto;border:1px solid var(--line);border-radius:8px;margin:18px 0}}.context,.source{{color:var(--muted);font-size:14px}}figure{{margin:24px 0}}img{{display:block;width:100%;height:auto}}figcaption{{font-size:12px;color:var(--muted);margin:10px 0}}details{{border-top:1px solid var(--line);padding:15px 0 4px;margin:12px 0}}summary{{cursor:pointer;font-weight:600;font-size:14px}}details p,details li{{font-size:14px;color:#42576a}}details ul{{padding-left:22px}}details li{{margin:9px 0}}.downloads{{margin-top:22px}}.downloads h3{{font-size:14px}}.downloads a{{display:inline-block;font-size:12px;padding:6px 10px;border:1px solid var(--line);border-radius:5px;margin:3px 6px 3px 0;text-decoration:none}}.downloads a:hover{{background:#edf3f8}}.video-card{{border-top:1px solid var(--line);padding:24px 0 4px;margin-top:22px;scroll-margin-top:20px}}.video-heading{{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}}.video-heading h3{{margin:0}}video{{display:block;width:100%;height:auto;background:#101722;border-radius:8px;margin:16px 0}}.video-meta{{font-size:12px;color:var(--muted);gap:12px 24px}}.video-nav{{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}}.video-nav a{{padding:5px 12px;border:1px solid var(--line);border-radius:5px;text-decoration:none;font-size:12px}}footer{{border-top:1px solid var(--line);padding:20px 0;color:var(--muted);font-size:12px}}@media(max-width:650px){{main{{padding:22px 14px}}section{{padding:20px 16px}}.metrics{{grid-template-columns:repeat(2,1fr);gap:10px}}.metric{{padding:13px}}.metric strong{{font-size:27px}}h2{{font-size:21px}}nav{{gap:10px 18px}}}}
{CSS}
</style></head><body><main><header id="overview"><span class="eyebrow">CSE 6367 Â· Computer Vision</span><h1>Retail Shoplifting Detection</h1><p class="subtitle">Experimental results and video analysis</p></header>
<nav aria-label="Report sections"><a href="#supervised-clips">Classification results</a><a href="#video-analysis">Video analysis</a><a href="#failure-audit">Error analysis</a><a href="#methodology">Methodology</a></nav>
<div class="metrics" aria-label="Held-out clip classification metrics">{overview_metrics}</div>
{supervised_section}
<section id="video-analysis"><span class="eyebrow">Experiment 01</span><h2>Shoplifting and normal activity predictions</h2><p>Choose a video to see the predicted activity, tracked people and exact alert times. Use the activity table to jump to a moment in the footage. Detailed model values remain available under Technical diagnostics.</p>{activity_overview}<div class="video-nav" aria-label="Video selection">{video_links}</div>{''.join(cards)}</section>
{failure_section}
<section id="methodology"><span class="eyebrow">Processing and evaluation</span><h2>Tracking and camera motion</h2><p>Camera movement in RV002 and RV007 was isolated before pose smoothing and temporal windowing. Scores expire at view boundaries. Accepted windows changed from 64 to 60 in RV002 and from 58 to 54 in RV007; the five remaining clips retained their original counts.</p><figure><img src="{image_uri}" alt="Pose score timelines before and after camera motion handling"><figcaption>Figure 3. Original and camera-aware pose score timelines.</figcaption></figure>
<details><summary>Visual review protocol and label status</summary><p>Source timelines were sampled at 2 frames per second: 433 inspected source frames and 52 skeleton-overlay samples, with denser inspection around camera movement. Observations were AI-assisted. All seven action classifications remain uncertain and are excluded from supervised training. Overhead views show fragmented tracks and missing joints; pose support alone does not establish an action label.</p><p>Reconstruction errors and joint confidence values are not theft probabilities. Fewer alerts or reduced scoring coverage does not establish higher detection accuracy.</p></details></section>
<footer>CSE 6367 Computer Vision Â· Retail Shoplifting Detection Â· Experimental results</footer>
</main>{JAVASCRIPT}</body></html>'''
    current=folder/'index.html'
    if current.exists() and not (folder/'review_form.html').exists():
        (folder/'review_form.html').write_bytes(current.read_bytes())
    current.write_text(page,encoding='utf-8')
    print('Results dashboard:',current)


if __name__=='__main__':main()
