$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

$DatasetDir = if ($args.Count -ge 1) { $args[0] } else { ".\data\ucf_crime" }
$VlmName = if ($args.Count -ge 2) { $args[1] } else { "videollama3" }
$ScoresDir = Join-Path $DatasetDir "scores\$VlmName"
$CaptionsDir = Join-Path $DatasetDir "captions\$VlmName"

$Required = @(
    (Join-Path $DatasetDir "frames"),
    (Join-Path $DatasetDir "annotations\test.txt"),
    (Join-Path $DatasetDir "annotations\Temporal_Anomaly_Annotation_for_Testing_Videos.txt"),
    $ScoresDir
)

foreach ($Path in $Required) {
    if (-not (Test-Path -LiteralPath $Path)) {
        Write-Error "Missing required path: $Path. Download/extract the official annotation package and raw videos before running full evaluation."
    }
}

$MetricsDir = Join-Path $ScoresDir "metrics"
New-Item -ItemType Directory -Force -Path $MetricsDir | Out-Null
python -m src.eval `
    --root_path (Join-Path $DatasetDir "frames") `
    --annotationfile_path (Join-Path $DatasetDir "annotations\test.txt") `
    --scores_dir $ScoresDir `
    --captions_dir $CaptionsDir `
    --output_dir $MetricsDir `
    --frame_interval 16 `
    --temporal_annotation_file (Join-Path $DatasetDir "annotations\Temporal_Anomaly_Annotation_for_Testing_Videos.txt") `
    --video_fps 30
