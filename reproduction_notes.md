# URF-HVAA Reproduction Notes

Date: 2026-07-02

## Repository

Reference repository cloned from `https://github.com/Rathgrith/URF-HVAA` into this workspace subdirectory.

Main entry points:

- Caption extraction: `python ./src/video_pre_caption.py`
- First-round VAD scoring: `bash scripts/query_llm_vad.sh`, module `src.llm_anomaly_scorer`
- Suspicious-window extraction: `python ./src/score_filter.py`
- Suspicious phrase extraction: `python ./src/summarize_window.py`
- Score refinement: `bash scripts/refine_score.sh`, module `src.refine_with_tag`
- VAD evaluation: `python -m src.eval` or `scripts/eval_{ucf,xd,ub,msad}.sh`
- VAL: `python src/val_priors.py`
- VAU: `python src/vau_priors.py`
- VAU text metrics: `python src/compute_bleu.py <ground_truth.json> <predictions.json>`
- GPT-based VAU scoring: `python src/gpt_score_eval.py --pred_path ... --output_path ...`

## Environment

Official setup:

```bash
conda env create -f environment.yml
conda activate VAA
```

The official environment targets Python 3.10, PyTorch 2.5, CUDA-capable Linux-style packages, and heavy model dependencies. On this Windows workstation, a full install is expected to need extra care because `faiss-gpu`, `flash-attn`, `triton`, and some CUDA wheels are platform/compiler sensitive. I did not silently change those major packages.

Smoke-demo environment used here:

```powershell
python --version
python -c "import numpy, sklearn, scipy, PIL; print('imports ok')"
powershell -ExecutionPolicy Bypass -File .\run_demo.ps1
```

Observed Python: 3.12.4. Required imports for the synthetic smoke demo were available.

## Model and API Requirements

Main baseline model requirements:

- Video caption backbone: `DAMO-NLP-SG/VideoLLaMA3-7B`, loaded by `src/video_pre_caption.py`.
- Text scoring/refinement model: Llama 3.1 8B Instruct original checkpoint, expected at `./libs/llama/llama3.1-8b/`.
- Required Llama files include `consolidated.00.pth`, `params.json`, and `tokenizer.model`.
- README expects `consolidated.00.pth` SHA256 `ab33d910f405204e5d388bc3521503584800461dc96808e287821dd451c1edac`.

API keys:

- `OPENAI_API_KEY` is required only by `src/gpt_score_eval.py` for GPT-4.1-based VAU scoring.
- `HF_TOKEN` is optional but likely needed for gated Hugging Face downloads.
- See `.env.example`.

No OpenAI/Gemini API is used in the main VAD scoring path. OpenAI is only used for one optional generative-output evaluation script.

## Data Requirements

The README points to a Google Drive package containing preprocessed annotations, captions, scores, refined scores, and outputs. Raw videos must be obtained from each dataset's original release.

Expected dataset layout:

```text
./data/
  {dataset_name}/
    annotations/
    videos/
      {video_basename}.mp4
    frames/
      {video_basename}/
        000001.jpg
        000002.jpg
    captions/
    scores/
    refined_scores/
```

Supported/evident datasets from scripts:

- UCF-Crime: `data/ucf_crime`
- XD-Violence: `data/xd_violence`
- UBNormal: `data/UBNormal`
- MSAD: `data/MSAD`

The project background mentions ShanghaiTech, Avenue, and Ped2, but this repository's shipped evaluation scripts only expose UCF-Crime, XD-Violence, UBNormal, and MSAD.

Annotation format:

- `annotations/test.txt`: rows parsed by `VideoRecord` as `{video_or_frame_dir} {start_frame} {end_frame} {label_id_or_ids}`.
- Temporal anomaly annotations: rows parsed as `{video_name} ... {start end pairs, -1 padded}`.
- Score JSON: one file per video, mapping frame index strings to numeric anomaly scores, for example `{"0": 0.1, "16": 0.8}`.
- Caption JSON: one file per video, mapping frame index strings to per-clip captions.

## Evaluation Metrics

`src/eval.py` computes:

- Frame-level ROC AUC
- Frame-level PR AUC
- Optimal ROC threshold by Youden's J
- Optimal PR/F1 threshold
- Max F1 printed to stdout

`src/compute_bleu.py` computes VAU text metrics:

- BLEU-1/2/3/4 and BLEU sum
- CIDEr
- METEOR
- ROUGE-L

`src/gpt_score_eval.py` computes GPT-based Reasonability, Detail, and Consistency scores.

## Minimal Demo Run

Because the official demo is not provided and the required datasets/model checkpoints are absent, I added a minimal synthetic smoke demo:

- `src/minimal_demo.py`
- `run_demo.ps1`
- `run_demo.sh`

Run:

```powershell
cd T:\Bigwork\SMILES.URF-HVAA\URF-HVAA
powershell -ExecutionPolicy Bypass -File .\run_demo.ps1
```

Output:

```text
outputs/minimal_demo/minimal_demo_result.json
outputs/minimal_demo/synthetic_dataset/...
```

Observed result:

- ROC AUC: `1.0`
- PR AUC: `1.0`
- Best F1: approximately `1.0`
- Recovered suspicious interval is saved to `outputs/minimal_demo/synthetic_dataset/scores/mock_videollama3/highest_lowest_intervals.json`.

This is not a paper-result reproduction. It is a smoke test that verifies the local scoring/evaluation output formats and the suspicious-window logic without requiring VideoLLaMA3, Llama3.1, or the official datasets.

## Full Baseline Commands

After downloading data and model weights:

```bash
conda env create -f environment.yml
conda activate VAA
python ./src/video_pre_caption.py --video_folder "./data/ucf_crime/videos/" --index_file "./data/ucf_crime/annotations/test.txt" --output_dir "./data/ucf_crime/captions/videollama3_json_results" --interval 10
bash scripts/query_llm_vad.sh
python ./src/score_filter.py
python ./src/summarize_window.py
bash scripts/refine_score.sh
bash scripts/eval_ucf.sh
```

If precomputed captions/scores from the Google Drive package are available, captioning and first-round scoring can be skipped as described in the README.

I also added guarded wrappers:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_eval.ps1
```

```bash
bash run_eval.sh
```

These currently report missing data paths until the official data package and raw frames are present.

## Current Status

Works now:

- Repository cloned.
- Structure, entry scripts, dependencies, model/API requirements, data format, and metrics inspected.
- `.env.example` created.
- Minimal synthetic smoke demo runs and writes outputs under `outputs/minimal_demo/`.
- Evaluation wrapper scripts added with clear missing-data checks.

Not yet reproduced:

- Official VAD numbers from the paper.
- VideoLLaMA3 caption extraction.
- Llama3.1 8B first-round scoring/refinement.
- Full UCF-Crime/XD/UB/MSAD evaluation.

Missing next:

- Official Google Drive annotation/precomputed package extracted under `./data`.
- Raw videos and extracted frames for target dataset.
- Llama3.1 8B Instruct original checkpoint under `./libs/llama/llama3.1-8b/`.
- A platform-compatible full environment, preferably Linux/WSL2 or a CUDA Linux machine for `flash-attn`/`faiss-gpu`.
