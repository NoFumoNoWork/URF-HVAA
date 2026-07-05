# Data Inventory

## XD-Violence

- Annotation exists: True (`data\xd_violence\annotations\temporal_anomaly_annotation_for_testing_videos.txt`)
- Annotation videos: 800
- Multi-anomaly videos: 260
- Videos dir exists: False (0 entries)
- Frames dir exists: True (0 video frame dirs)
- Captions dir exists: True (800 json files)
- Matched annotation videos with any score: 800
- Missing score videos: 0

Score dirs:
- `data\xd_violence\refined_scores\videollama3`: exists=True, json=800, matched=800
- `data\xd_violence\scores\videollama3`: exists=False, json=0, matched=0

Can do:
- temporal coverage analysis
- score curve analysis
- top-k interval evaluation

Cannot do:
- raw mp4 visual inspection
- frame-level visual verification for all videos

## UCF-Crime

- Annotation exists: True (`data\ucf_crime\annotations\Temporal_Anomaly_Annotation_for_Testing_Videos.txt`)
- Annotation videos: 290
- Multi-anomaly videos: 16
- Videos dir exists: False (0 entries)
- Frames dir exists: False (0 video frame dirs)
- Captions dir exists: True (290 json files)
- Matched annotation videos with any score: 290
- Missing score videos: 0

Score dirs:
- `data\ucf_crime\scores\videollama3`: exists=True, json=290, matched=290
- `data\ucf_crime\refined_scores\videollama3`: exists=True, json=290, matched=290

Can do:
- temporal coverage analysis
- score curve analysis
- top-k interval evaluation

Cannot do:
- raw mp4 visual inspection
- frame-level visual verification for all videos
