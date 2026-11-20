"""Publish read-only review findings, replacing the request for user annotation."""
import base64
import html
import hashlib
import json
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


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
        failure_section=f'<section id="failure-audit"><h2>Label and failure audit</h2><ul>{items}</ul><img src="{plot_uri}" alt="Frozen model recall and positive windows missed by all six models"><details><summary>Audit data and provenance</summary><p>'+ ' · '.join(downloads)+'</p></details></section>'
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
        cards.append(f'''<section><h2>{escape(case['case_id'])} · Retail scene</h2>
<p>{escape(note['notes'])}</p><p><strong>Action classification: uncertain.</strong> Tracking assessment: {escape(note['tracking_quality'])}.</p>
<video controls preload="metadata" poster="{escape(poster_url)}" src="{escape(video_url)}"></video>
<p>{video['decoded_output_frames']:,} verified frames · {video['output_fps']:g} FPS · {video['unverified_alert_frames']} frames with unverified model alerts.</p>
<details><summary>Original video and model diagnostics</summary><video controls preload="none" src="{escape(case['source_file'])}"></video>
<div class="scroll"><table><tr><th>Model</th><th>Processing</th><th>Accepted windows</th><th>Score coverage</th><th>Flagged windows</th></tr>{''.join(rows)}</table></div></details></section>''')
    image=ROOT/analysis['plots'][0]
    image_uri='data:image/png;base64,'+base64.b64encode(image.read_bytes()).decode()
    page=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>Autonomous retail review results</title><style>
body{{margin:0;background:#f3f6fb;color:#182536;font:16px/1.55 system-ui,sans-serif}}main{{max-width:1280px;margin:auto;padding:28px}}h1{{font-size:30px;margin-bottom:8px}}h2{{font-size:21px}}section,.summary{{background:white;padding:22px;border:1px solid #dce3ef;border-radius:10px;margin:18px 0}}.status{{color:#185a47;font-weight:650}}.note{{padding:14px;background:#fff7e6;border-left:4px solid #ba7e13}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{text-align:left;border-bottom:1px solid #dce3ef;padding:8px}}video{{display:block;width:100%;background:#101722;margin:14px 0}}img{{width:100%}}summary{{cursor:pointer;font-weight:600}}.scroll{{overflow-x:auto}}
</style></head><body><main><h1>Autonomous retail review results</h1><p class="status">Review and local model analysis completed. No annotation task is assigned to you.</p>
<div class="summary"><strong>Seven retail scenes reviewed.</strong> The source timelines were sampled at 2 frames/second (433 inspected frames), with skeleton samples and denser camera-motion inspection. These are AI observations, not independently verified action labels.</div>
<p class="note">The samples do not establish theft or the absence of theft. All seven action classifications remain uncertain and are excluded from supervised training. Original RetailS benchmark results remain unchanged.</p>
{failure_section}
<section><h2>Annotated model videos</h2><p>Each video displays person IDs, skeletons, joint visibility and confidence, window timing, six A/B model scores and their thresholds. The declared primary model is A seed 0. A red banner marks possible shoplifting / suspicious activity when its score reaches the threshold; this is an unverified model alert. Scores are reconstruction errors, not theft probabilities. Gray periods have no eligible score or paused scoring during camera movement. Benchmark metrics shown in the panel describe held-out RetailS results, not the accuracy of the playing video.</p></section>
<section><h2>Completed engineering change</h2><p>RV002 contains a digital zoom and RV007 a camera pan. The new scoring path isolates these intervals before pose smoothing and temporal windowing, and expires scores at view boundaries. Frozen A/B models were re-evaluated for all three seeds.</p>
<p>Usable windows changed from 64 to 60 in RV002 and 58 to 54 in RV007. The five other clips retained their original window counts. Reduced coverage or fewer flags is not evidence of higher detection accuracy.</p><img src="{image_uri}" alt="Original and camera-aware A/B score timelines for the two clips with camera movement"></section>
{''.join(cards)}<section><h2>Interpretation</h2><p>Overhead views have fragmented tracks and missing joints. Stable upper-body tracks can still lack the product, hand or bag evidence needed to recognize an action. The current observations support technical diagnosis, not accuracy claims or new theft labels.</p></section>
</main></body></html>'''
    current=folder/'index.html'
    if current.exists() and not (folder/'review_form.html').exists():
        (folder/'review_form.html').write_bytes(current.read_bytes())
    current.write_text(page,encoding='utf-8')
    print('Read-only autonomous report:',current)


if __name__=='__main__':main()
