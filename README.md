# FT-Mini: Fine-tuned RDF Triple Extraction

Utilities to fine-tune **gpt-4o-mini** into **FT-Mini**, a model that extracts DBpedia-aligned RDF triples (Turtle) from plain text. Training data is compiled from WebNLG / MuSiQue / HotpotQA-style sources.

On **HotMusValid**, FT-Mini reached **F1 0.3236** vs **0.2375** for a 2-step prompting baseline in an external comparison study.

## Quick start

From the project root (`ft-mini/`):

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements-inference.txt
```

1. Copy and edit `inference_test/inference.yaml` (API key, model name, `input_texts`).
2. Run a smoke test:

```bash
python inference_test/run_inference_yaml.py
```

For batch inference and evaluation, install the matching requirement files and use the YAML runners described below.

## Project layout

```
ft-mini/
├── batch_test/              # OpenAI Batch API (large-scale inference)
├── code_for_finetuning/     # Upload JSONL and start fine-tuning jobs
├── datasets/                # Ground-truth JSON (gt_text_*.json)
├── evaluation/              # Parse batch output, score vs ground truth
├── inference_test/          # Single-request inference smoke tests
├── prompts/                 # Prompt fragments (p0.txt … p4.txt)
└── train_validation_sets/   # Build train/validation JSONL for fine-tuning
```

All paths in YAML configs are resolved **relative to this repository root**, not a parent monorepo.

## Configuration

Secrets and run parameters live in **YAML files next to the scripts that read them** (not a single root `config.yaml`):

| File | Used by |
|------|---------|
| `inference_test/inference.yaml` | `run_inference_yaml.py` |
| `batch_test/batch_test_run.yaml` | `run_batch_test_yaml.py` |
| `evaluation/evaluation.yaml` | `run_evaluation_yaml.py` |

Do not commit real API keys. Add `*api_key*` patterns or local overrides to `.gitignore` if you keep secrets in YAML.

Some legacy scripts still use placeholders (`your_key`, `your_model`). Prefer the YAML runners; they pass `OPENAI_API_KEY` and `MODEL_NAME` via environment variables where needed.

### Example: inference (`inference_test/inference.yaml`)

```yaml
api_key: "sk-..."
model: "ft:gpt-4o-mini-2024-07-18:personal::YOUR_MODEL_ID"
temperature: 0.0
system_prompt: |
  # Role: ...
input_texts:
  - "Example text to extract triples from."
```

## Dependencies

Install only what you need:

| Task | Requirements file |
|------|-------------------|
| Inference smoke test | `requirements-inference.txt` |
| Batch jobs | `requirements-batch.txt` |
| Evaluation (DBpedia Spotlight, SPARQL, embeddings) | `requirements-evaluation.txt` |

Fine-tuning upload/monitor scripts need `openai` (included in inference/batch requirements).

Evaluation additionally uses live **DBpedia Spotlight** and **SPARQL** endpoints; allow network access when running `evaluation/`.

## Data formats

### Fine-tuning JSONL (`train_set.jsonl`, `validation_set.jsonl`)

One JSON object per line, OpenAI chat format:

- `messages[].role`: `system`, `user`, `assistant`
- `assistant` content: target Turtle triples

Built by `train_validation_sets/set_creator/set_creator.py` from `train_compilation.json` and `validation_compilation.json` in the same folder.

### Batch request JSONL

Built by `batch_test/batch_creator.py`. Input JSON items need:

- `row_idx` → `custom_id`
- `text` → user message content

Set the model via `MODEL_NAME` or `OPENAI_MODEL_NAME` environment variable (the YAML batch runner sets this automatically).

### Evaluation inputs

- **Ground truth** (`datasets/gt_text_*.json`): `row_idx`, `triples` (list of `"subject , predicate , object"` strings)
- **Predictions** (after formatting): `custom_id`, `content` (list of triple strings)

`evaluation/formater_for_evaluation.py` converts OpenAI batch output JSONL into the predictions shape.

## Workflows

### 1. Build fine-tuning data

```bash
cd train_validation_sets/set_creator
python set_creator.py
```

Outputs `train_set.jsonl` and `validation_set.jsonl` in that directory. Edit `INPUT_TRAIN_JSON` / `INPUT_VAL_JSON` at the top of `set_creator.py` if your compilation filenames differ.

### 2. Fine-tune on OpenAI

1. Edit API key and JSONL paths in `code_for_finetuning/uploader_for_finetuning.py`.
2. Upload and start the job:

```bash
python code_for_finetuning/uploader_for_finetuning.py
```

3. Monitor and retrieve the fine-tuned model id:

```bash
python code_for_finetuning/process_monitor.py   # set job_id in script
python code_for_finetuning/download_model_name.py
```

### 3. Batch inference (recommended)

1. Edit `batch_test/batch_test_run.yaml` (`openai_api_key`, `model_name`, `texts_json`, output paths).
2. Run the full pipeline (create requests → submit → monitor → download):

```bash
python batch_test/run_batch_test_yaml.py
```

Or run steps manually:

```bash
set OPENAI_API_KEY=sk-...
set MODEL_NAME=ft:gpt-4o-mini-...
python batch_test/batch_creator.py datasets/gt_text_hotmusvalid.json batch_outputs/batch_requests.jsonl
python batch_test/create_batch_upload.py batch_outputs/batch_requests.jsonl
python batch_test/process_monitor.py <JOB_ID>
python batch_test/download_batch_file.py <OUTPUT_FILE_ID> batch_outputs/batch_results.jsonl
```

### 4. Evaluate predictions (recommended)

1. Place batch result JSONL at the path in `evaluation/evaluation.yaml` (or update the config).
2. Run format → dedupe → score:

```bash
python evaluation/run_evaluation_yaml.py
```

Manual equivalent:

```bash
python evaluation/formater_for_evaluation.py batch_outputs/batch_results.jsonl batch_outputs/predictions.json
python evaluation/duplicate_cleaner.py batch_outputs/predictions.json batch_outputs/predictions_clean.json
# Then call evaluate_triples from evaluation/evaluation.py or adjust its __main__ paths
```

### 5. Quick inference test

```bash
python inference_test/run_inference_yaml.py
# optional: python inference_test/run_inference_yaml.py --config path/to/inference.yaml
```

Legacy script: `inference_test/run_inference.py` (hardcoded key, model, and texts).

## Module reference

### `batch_test/`

| Script | Purpose |
|--------|---------|
| `run_batch_test_yaml.py` | End-to-end batch workflow from `batch_test_run.yaml` |
| `batch_creator.py` | Build batch request `.jsonl` from a texts JSON file |
| `create_batch_upload.py` | Upload JSONL and create batch job (`OPENAI_API_KEY`) |
| `process_monitor.py` | Poll batch job status |
| `download_batch_file.py` | Download completed batch output |

### `code_for_finetuning/`

| Script | Purpose |
|--------|---------|
| `uploader_for_finetuning.py` | Upload train/val JSONL and start fine-tuning |
| `process_monitor.py` | Poll fine-tuning job status |
| `download_model_name.py` | Print fine-tuned model name for a job |

### `datasets/`

Ground-truth JSON files for evaluation:

- `gt_text_webnlg.json`, `gt_text_webnlg_plus.json`, `gt_text_webnlg_plus_plus.json`
- `gt_text_hotmusvalid.json`

### `evaluation/`

| Script | Purpose |
|--------|---------|
| `run_evaluation_yaml.py` | Format, dedupe, and score from `evaluation.yaml` |
| `formater_for_evaluation.py` | Batch output JSONL → predictions JSON |
| `duplicate_cleaner.py` | Remove duplicate triples per `content` list |
| `evaluation.py` | Entity linking, sameAs checks, predicate similarity, P/R/F1 |

### `inference_test/`

| Script | Purpose |
|--------|---------|
| `run_inference_yaml.py` | Chat completions from `inference.yaml` |
| `run_inference.py` | Legacy hardcoded smoke test |

### `prompts/`

`p0.txt` … `p4.txt` → prompt templates for experimentation.

### `train_validation_sets/set_creator/`

| File | Purpose |
|------|---------|
| `set_creator.py` | Compilation JSON → OpenAI fine-tuning JSONL |
| `train_compilation.json`, `validation_compilation.json` | Source training examples |
