"""Displayed model alerts are unverified; a low score never confirms normality."""
import math

PRIMARY_RUN='expanded_a_seed0'


def active_window(windows,frame_id,stride):
    eligible=[w for w in windows if int(w['end_frame'])<=frame_id<min(int(w['end_frame'])+stride,int(w['scene_last_frame'])+1)]
    return max(eligible,key=lambda w:int(w['end_frame'])) if eligible else None


def model_status(scores,thresholds,camera_motion=False):
    if camera_motion:return 'CAMERA MOTION - SCORING PAUSED'
    primary=scores.get(PRIMARY_RUN)
    if primary is None:return 'UNSCORED - AWAITING ELIGIBLE WINDOW'
    if not math.isfinite(primary) or not math.isfinite(thresholds[PRIMARY_RUN]) or thresholds[PRIMARY_RUN]<=0:
        raise ValueError('Finite score and positive threshold required')
    if primary>=thresholds[PRIMARY_RUN]:return 'POSSIBLE SUSPICIOUS ACTIVITY - UNVERIFIED'
    return 'BELOW ANOMALY THRESHOLD - NOT A NORMALITY CLAIM'
