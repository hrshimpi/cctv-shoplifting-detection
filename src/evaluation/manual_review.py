"""Validate attributed observations; AI review is not independent ground truth."""


def validate_review(payload, manifest):
    if payload.get('schema')!=1 or payload.get('manifest_id')!=manifest['manifest_id']:
        raise ValueError('Review belongs to a different video manifest')
    if not isinstance(payload.get('reviewer'),str) or not payload['reviewer'].strip():
        raise ValueError('Reviewer name or identifier is required')
    reviewer_type=payload.get('reviewer_type','human_declared')
    if reviewer_type not in ['human_declared','ai_assisted']:
        raise ValueError('Unknown reviewer provenance')
    cases={case['case_id']:case for case in manifest['cases']}
    reviews=payload.get('reviews',[])
    if len(reviews)!=len(cases) or len({r.get('case_id') for r in reviews})!=len(cases):
        raise ValueError('Exactly one review entry per case is required')
    completed=normal=event_cases=uncertain=events=person_linked=0
    for review in reviews:
        case=cases.get(review.get('case_id'))
        if case is None or review.get('source_sha256')!=case['source_sha256']:
            raise ValueError('Unknown or changed source video')
        status=review.get('status','pending')
        if status not in ['pending','no_visible_anomaly','events_present','uncertain']:
            raise ValueError('Unknown review status')
        annotations=review.get('events',[])
        if status=='pending':
            if annotations:
                raise ValueError('Pending cases cannot contain event labels')
            continue
        if reviewer_type=='ai_assisted':
            evidence=review.get('evidence_frame_ids',[])
            if (review.get('review_method')!='sampled_frames' or review.get('reviewed_whole_video') is not False
                or not evidence or evidence!=sorted(set(evidence))
                or any(type(fid) is not int or not 0<=fid<case['frame_count'] for fid in evidence)
                or evidence[0]!=0 or evidence[-1]!=case['frame_count']-1):
                raise ValueError('AI sampled review must declare valid full-timeline sample evidence, not full-frame review')
            if status=='no_visible_anomaly':
                raise ValueError('Sampled AI review cannot establish absence of anomalies')
        elif review.get('reviewed_whole_video') is not True:
            raise ValueError('Complete-video review is required')
        if review.get('retail_scene') not in ['yes','no','uncertain']:
            raise ValueError('Complete-video review and retail-scene assessment are required')
        if review.get('tracking_quality') not in ['usable','fragmented','occluded','uncertain']:
            raise ValueError('Tracking-quality assessment is required')
        if status=='no_visible_anomaly' and annotations:
            raise ValueError('Normal review contradicts event annotations')
        if status=='events_present' and not any(e.get('label')=='observable_anomaly' for e in annotations):
            raise ValueError('Events-present status needs an observable-anomaly interval')
        if status=='uncertain' and any(e.get('label')!='uncertain' for e in annotations):
            raise ValueError('Uncertain cases cannot contain asserted anomaly labels')
        for event in annotations:
            start,end=event.get('start_frame'),event.get('end_frame')
            if type(start) is not int or type(end) is not int or not 0<=start<=end<case['frame_count']:
                raise ValueError('Event frames must be integers inside the source video')
            if event.get('label') not in ['observable_anomaly','uncertain']:
                raise ValueError('Unsupported observation label')
            if event.get('confidence') not in ['high','medium','low'] or not isinstance(event.get('observation'),str) or not event['observation'].strip():
                raise ValueError('Describe the observed action and confidence')
            person=event.get('track_id')
            if person is not None and str(person) not in case['track_ids']:
                raise ValueError('Track ID is absent from this pose export')
            events+=1; person_linked+=int(person is not None)
        completed+=1; normal+=int(status=='no_visible_anomaly')
        event_cases+=int(status=='events_present'); uncertain+=int(status=='uncertain')
    return dict(cases=len(cases),completed_cases=completed,pending_cases=len(cases)-completed,
        no_visible_anomaly_cases=normal,events_present_cases=event_cases,uncertain_cases=uncertain,
        annotated_intervals=events,person_linked_intervals=person_linked,
        complete=completed==len(cases),reviewer_type=reviewer_type,verified_ground_truth=False,
        eligible_for_supervised_training=False,
        role='AI observations only; not reference labels' if reviewer_type=='ai_assisted' else 'previously inspected pilot clips; not an independent confirmation set')
