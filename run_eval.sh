#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

dataset_dir="${1:-./data/ucf_crime}"
vlm_name="${2:-videollama3}"
scores_dir="${dataset_dir}/scores/${vlm_name}"
captions_dir="${dataset_dir}/captions/${vlm_name}"

required=(
  "${dataset_dir}/frames"
  "${dataset_dir}/annotations/test.txt"
  "${dataset_dir}/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
  "${scores_dir}"
)

for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    echo "Download/extract the official annotation package and raw videos before running full evaluation." >&2
    exit 2
  fi
done

mkdir -p "${scores_dir}/metrics"
python -m src.eval \
  --root_path "${dataset_dir}/frames" \
  --annotationfile_path "${dataset_dir}/annotations/test.txt" \
  --scores_dir "${scores_dir}" \
  --captions_dir "${captions_dir}" \
  --output_dir "${scores_dir}/metrics" \
  --frame_interval 16 \
  --temporal_annotation_file "${dataset_dir}/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt" \
  --video_fps 30
