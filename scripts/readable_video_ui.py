"""Plain-language playback summaries and seekable model-prediction timelines."""
import html
import json
import shutil
from pathlib import Path
from scripts.prepare_mnnit_clips import sha256


def esc(value):return html.escape(str(value))


def clock(value):return f'{int(value)//60:02d}:{value%60:05.2f}'


def badge(state,label):return f'<span class="activity-badge state-{esc(state)}">{esc(label)}</span>'


def load_report(root,manifest):
    path=root/'outputs/readable_video_results/report.json'
    if not path.exists():return None
    report=json.loads(path.read_text())
    if not report['complete'] or report['manifest_id']!=manifest['manifest_id']:
        raise ValueError('Readable video report is incomplete or stale')
    cases={r['case_id']:r for r in report['cases']}
    if set(cases)!={c['case_id'] for c in manifest['cases']}:raise ValueError('Readable video set is incomplete')
    for case in manifest['cases']:
        r=cases[case['case_id']]
        if r['source_sha256']!=case['source_media_sha256'] or r['decoded_output_frames']!=case['frame_count']:
            raise ValueError('Readable source/count differs from the manifest')
        if sha256(root/'outputs/manual_validation'/r['video'])!=r['video_sha256']:
            raise ValueError('Readable video changed')
        if sha256(root/'outputs/video_analysis'/r['case_id']/'frame_values.jsonl')!=r['frame_values_sha256']:
            raise ValueError('Readable timeline per-frame evidence changed')
    return cases


def overview(cases):
    rows=[]
    for cid,r in cases.items():
        state='alert' if r['alert_periods'] else 'normal'
        label='Possible shoplifting' if r['alert_periods'] else 'Normal activity predicted'
        rows.append(f'<tr><th scope="row"><a href="#{cid.lower()}">{cid}</a><small>{esc(r["scene"])}</small></th><td>{clock(r["duration_seconds"])}</td><td>{badge(state,label)}</td><td>{r["alert_periods"]}</td><td>{r["alert_seconds"]:.2f}s</td><td>{r["assessment_coverage"]:.1%}</td></tr>')
    return f'''<div class="activity-legend">{badge('alert','Possible shoplifting')}{badge('normal','Normal activity predicted')}{badge('unknown','Unable to assess')}</div>
<p class="context">These are model predictions. Normal means no alert for the assessed people; it does not prove that shoplifting was absent. Unable to assess means camera movement or insufficient tracking. Each alert is a point to review, not a confirmed theft.</p>
<div class="scroll"><table class="activity-overview"><caption>Video summaries based on model outputs</caption><thead><tr><th>Video</th><th>Duration</th><th>Overall model result</th><th>Alert periods</th><th>Alert duration</th><th>Assessed</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>'''


def card(root,case,note,r,old_video_url,diagnostic_rows):
    cid=case['case_id'];player=cid.lower()+'-activity'
    out=root/'outputs/readable_video_results';media=root/'outputs/manual_validation/media'
    for suffix in ['timeline.json','timeline.csv']:
        shutil.copy2(out/f'{cid}_{suffix}',media/f'{cid}_{suffix}')
    table=[]
    for row in r['seconds']:
        people=', '.join('P'+str(p) for p in row['people']) or '—'
        spans='<br>'.join(f'{clock(s["start_seconds"])}–{clock(s["end_seconds_exclusive"])}' for s in row['alert_spans'])
        if not spans:spans='No alert in assessed frames' if row['assessed_frames'] else 'Not assessed'
        table.append(f'''<tr data-state="{row['state']}" data-start="{row['start_seconds']}" data-end="{row['end_seconds_exclusive']}"><td class="time-range">{clock(row['start_seconds'])}–{clock(row['end_seconds_exclusive'])}</td><td>{badge(row['state'],row['label'])}</td><td>{esc(people)}</td><td title="{row['assessed_frames']} of {row['frames']} frames assessed">{row['assessed_frames']/row['frames']:.0%}</td><td>{spans}</td><td><button type="button" class="seek-button" data-video="{player}" data-seek="{row['seek_seconds']}" aria-label="Play {cid} from {clock(row['seek_seconds'])}">Play</button></td></tr>''')
    alerts=[]
    for i,row in enumerate(r['alert_intervals'],1):
        people=', '.join('Person '+str(p) for p in row['people'])
        alerts.append(f'<li><button type="button" class="seek-button" data-video="{player}" data-seek="{row["start_seconds"]}">{clock(row["start_seconds"])}–{clock(row["end_seconds_exclusive"])}</button> <span>{esc(people)} · {row["duration_seconds"]:.2f}s</span></li>')
    alert_list='<ol class="alert-list">'+''.join(alerts)+'</ol>' if alerts else '<p>No model-alert period was recorded.</p>'
    observations=''
    for event in note.get('events',[]):
        observations+=f'<li>{clock(event["start_frame"]/r["fps"])}–{clock((event["end_frame"]+1)/r["fps"])}: {esc(event["observation"])}</li>'
    if observations:observations='<details><summary>Sampled visual observations</summary><p>These observations are separate from model predictions and do not establish an action label.</p><ul>'+observations+'</ul></details>'
    payload=[{k:segment[k] for k in ['state','label','reason','start_seconds','end_seconds_exclusive','people']} for segment in r['exact_intervals']]
    data=json.dumps(dict(duration=r['duration_seconds'],intervals=payload),separators=(',',':')).replace('<','\\u003c')
    return f'''<article class="video-card readable-card" id="{cid.lower()}"><div class="video-heading"><h3>{cid} · {esc(r['scene'])}</h3><a class="back-link" href="#video-analysis">All videos</a></div>
<p>{esc(r['scene_summary'])}</p>
<div class="playback-summary"><span data-live-label class="activity-badge state-unknown">Unable to assess</span><span data-live-time>00:00.00</span><span class="context">Model prediction at playback time</span></div>
<video id="{player}" class="readable-main" data-case="{cid}" controls preload="metadata" poster="{esc(r['poster'])}" src="{esc(r['video'])}"></video>
<div class="conclusion"><h4>Video conclusion</h4><p>{esc(r['conclusion'])}</p></div>
<div class="video-meta"><span>{r['alert_periods']} alert periods</span><span>{r['alert_seconds']:.2f}s flagged</span><span>{r['assessment_coverage']:.1%} assessed</span><span>{r['unknown_seconds']:.2f}s unable to assess</span></div>
<h4>Activity timeline</h4><p class="context">Each row covers up to one second. The alert-time column gives the exact flagged portions within that row. P1 means tracked Person 1; IDs are estimates.</p>
<div class="timeline-controls" role="group" aria-label="Filter {cid} activity timeline"><button type="button" data-filter="all" aria-pressed="true">All activity</button><button type="button" data-filter="alert" aria-pressed="false">Shoplifting alerts</button><button type="button" data-filter="normal" aria-pressed="false">Normal predictions</button><button type="button" data-filter="unknown" aria-pressed="false">Unable to assess</button></div>
<div class="scroll timeline-scroll"><table class="timeline"><thead><tr><th>Time range</th><th>Model prediction</th><th>Person IDs</th><th>Assessed</th><th>Exact alert time</th><th>Playback</th></tr></thead><tbody>{''.join(table)}</tbody></table></div><p class="filter-empty" hidden>No intervals match this filter.</p>
<details><summary>Exact shoplifting-alert periods</summary>{alert_list}<p>These are model-alert times, not verified action start/end annotations. End times are exclusive.</p></details>
{observations}
<details><summary>Technical diagnostics and original footage</summary><p>Detailed values for all six pose models are retained in the diagnostic video. The simple view uses the same A seed 0 scores, thresholds and camera handling; no new classifier has been applied to these videos.</p><video controls preload="none" src="{esc(old_video_url)}"></video><p>{esc(note['notes'])}</p><div class="scroll"><table><thead><tr><th>Model</th><th>Processing</th><th>Windows</th><th>Coverage</th><th>Flags</th></tr></thead><tbody>{diagnostic_rows}</tbody></table></div><video controls preload="none" src="{esc(case['source_file'])}"></video></details>
<div class="downloads"><a download href="media/{cid}_timeline.csv">Activity timeline (CSV)</a><a download href="media/{cid}_timeline.json">Full interval data (JSON)</a></div>
<script type="application/json" class="activity-data">{data}</script></article>'''


CSS='''
.activity-legend,.timeline-controls,.playback-summary{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin:15px 0}.activity-badge{display:inline-block;padding:5px 10px;border-radius:5px;font-size:12px;font-weight:650;white-space:normal;line-height:1.35}.state-alert{background:#fff0db;color:#8b4500}.state-normal{background:#e6f3eb;color:#185a38}.state-unknown{background:#edf0f4;color:#4b5d71}.activity-overview th small{display:block;font-weight:400;font-size:11px;color:var(--muted)}.activity-overview td,.activity-overview th{text-align:left}.conclusion{border:1px solid #d9e4ed;background:#f4f8fb;border-radius:8px;padding:16px 18px;margin:15px 0}.conclusion h4{margin:0 0 6px;font-size:15px}.conclusion p{margin:0;font-size:14px}.readable-card h4{font-size:16px;margin:20px 0 8px}.back-link{font-size:12px}.playback-summary{padding:10px 12px;background:#f4f7fa;border-radius:7px;font-size:13px}.playback-summary [data-live-time]{font-variant-numeric:tabular-nums;font-weight:650}.timeline-scroll{max-height:420px}.timeline th,.timeline td{text-align:left;font-size:12px;white-space:normal;padding:10px}.timeline th{position:sticky;top:0;z-index:1}.timeline .time-range{white-space:nowrap}.timeline .current-interval{background:#edf5ff;box-shadow:inset 3px 0 #3974a1}.timeline-controls button,.seek-button{border:1px solid #cbd9e5;background:white;color:#24577e;border-radius:5px;padding:6px 10px;font:inherit;font-size:12px;cursor:pointer}.timeline-controls button[aria-pressed=true]{background:#24577e;color:white;border-color:#24577e}.seek-button:hover{background:#eaf2f9}button:focus-visible{outline:3px solid #83aac8;outline-offset:2px}.alert-list{padding-left:20px}.alert-list li{margin:10px 0}.alert-list span{font-size:13px}.filter-empty{font-size:13px;color:var(--muted)}
'''

JAVASCRIPT='''
<script>
document.querySelectorAll('.readable-card').forEach(card=>{
 const video=card.querySelector('.readable-main');
 const data=JSON.parse(card.querySelector('.activity-data').textContent);
 const badge=card.querySelector('[data-live-label]');
 const time=card.querySelector('[data-live-time]');
 const rows=[...card.querySelectorAll('.timeline tbody tr')];
 const clock=s=>String(Math.floor(s/60)).padStart(2,'0')+':'+(s%60).toFixed(2).padStart(5,'0');
 function update(){
  const t=Math.min(video.currentTime,Math.max(0,data.duration-0.000001));
  const interval=data.intervals.find(r=>r.start_seconds<=t&&t<r.end_seconds_exclusive);
  if(interval){badge.textContent=interval.label;badge.className='activity-badge state-'+interval.state;}
  time.textContent=clock(video.currentTime);
  rows.forEach(row=>row.classList.toggle('current-interval',Number(row.dataset.start)<=t&&t<Number(row.dataset.end)));
 }
 video.addEventListener('timeupdate',update);video.addEventListener('seeked',update);video.addEventListener('loadedmetadata',update);update();
 card.querySelectorAll('button[data-filter]').forEach(button=>button.addEventListener('click',()=>{
  const state=button.dataset.filter;let visible=0;
  rows.forEach(row=>{row.hidden=state!=='all'&&row.dataset.state!==state;if(!row.hidden)visible++;});
  card.querySelectorAll('button[data-filter]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
  card.querySelector('.filter-empty').hidden=visible!==0;
 }));
});
document.addEventListener('click',event=>{
 const button=event.target.closest('button[data-seek]');if(!button)return;
 const video=document.getElementById(button.dataset.video);const seconds=Number(button.dataset.seek);
 document.querySelectorAll('video').forEach(other=>{if(other!==video)other.pause();});
 function play(){video.currentTime=Math.max(0,Math.min(seconds,video.duration||seconds));video.play().catch(()=>{});}
 if(video.readyState>=1)play();else{video.addEventListener('loadedmetadata',play,{once:true});video.load();}
 video.scrollIntoView({behavior:'smooth',block:'center'});
});
</script>
'''
