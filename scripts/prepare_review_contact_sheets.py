"""Extract traceable visual samples; decoding a frame does not mean it was reviewed."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view',choices=['source','overlay'],default='source')
    parser.add_argument('--sample-fps',type=float,default=2)
    parser.add_argument('--cases',nargs='+')
    args=parser.parse_args()
    folder=ROOT/'outputs/manual_validation'
    manifest=json.loads((folder/'review_manifest.json').read_text())
    destination=folder/'visual_samples'/args.view; destination.mkdir(parents=True,exist_ok=True)
    records=[]
    for case in manifest['cases']:
        if args.cases and case['case_id'] not in args.cases: continue
        step=max(1,round(case['fps']/args.sample_fps))
        selected=set(range(0,case['frame_count'],step))|{case['frame_count']-1}
        cap=cv2.VideoCapture(str(folder/case[args.view+'_file']))
        samples=[];fid=0
        while True:
            ok,frame=cap.read()
            if not ok: break
            if fid in selected: samples.append((fid,frame.copy()))
            fid+=1
        cap.release()
        if fid!=case['frame_count']: raise ValueError('Source decode count changed')
        pages=[]
        for page,start in enumerate(range(0,len(samples),48),1):
            group=samples[start:start+48]
            rows=(len(group)+5)//6
            sheet=np.full((rows*266,6*320,3),245,dtype=np.uint8)
            for index,(frame_id,frame) in enumerate(group):
                y,x=(index//6)*266,(index%6)*320
                sheet[y:y+240,x:x+320]=frame
                cv2.putText(sheet,f"{case['case_id']} f{frame_id}  {frame_id/case['fps']:.2f}s",(x+5,y+258),cv2.FONT_HERSHEY_SIMPLEX,.46,(20,20,20),1,cv2.LINE_AA)
            path=destination/f"{case['case_id']}_{page:02d}.jpg"
            if not cv2.imwrite(str(path),sheet,[cv2.IMWRITE_JPEG_QUALITY,94]): raise ValueError('Cannot save sheet')
            pages.append(str(path.relative_to(ROOT)))
        records.append(dict(case_id=case['case_id'],view=args.view,sample_frame_ids=sorted(selected),
            source_frames=fid,sampled_frames=len(samples),pages=pages,
            coverage='Full timeline sampled; intervening frames were decoded but not visually reviewed'))
    (destination/'samples.json').write_text(json.dumps(records,indent=2))
    print(json.dumps([dict(case_id=r['case_id'],samples=r['sampled_frames'],pages=r['pages']) for r in records],indent=2))


if __name__=='__main__': main()
