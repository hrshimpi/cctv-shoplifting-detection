"""Split source tracks BEFORE interpolation/smoothing around verified camera motion."""
from dataclasses import replace


def split_track(track,excluded_intervals):
    intervals=sorted((int(start),int(end)) for start,end in excluded_intervals)
    if any(start<0 or end<start for start,end in intervals):
        raise ValueError('Invalid scene-motion exclusion interval')
    if any(intervals[i][0]<=intervals[i-1][1] for i in range(1,len(intervals))):
        raise ValueError('Scene-motion exclusions must not overlap')
    groups={}
    for frame_id,frame in sorted(track.frames.items()):
        if any(start<=frame_id<=end for start,end in intervals): continue
        scene=sum(frame_id>end for start,end in intervals)
        groups.setdefault(scene,{})[frame_id]=frame
    return [(scene,replace(track,frames=frames)) for scene,frames in sorted(groups.items())]
