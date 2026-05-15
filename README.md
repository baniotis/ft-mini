# Fine-tuning Pipeline for DBpedia RDF Triple Extraction (FT-Mini)

This project contains utilities to fine-tune `gpt-4o-mini` into the **FT-Mini** model, which generates DBpedia-aligned RDF triples (Turtle format) from input texts. Training data comes from WebNLG-MuSiQue-HotpotQA-style datasets. The resulting fine-tuned model is evaluated in the sibling folder `../Deliverables/Deliverable D6/hipporag_vs_ft_mini/`, where it outperformed a 3-step prompting baseline (F1: 0.3236 vs 0.2375) on the HotMusValid dataset.

## Repository Layout

- `batch_test/`: prepare and run OpenAI "batch" jobs for inference.
  - `batch_creator.py`: builds an OpenAI batch request `.jsonl` from a JSON list of texts (expects `row_idx` and `text` fields per entry). Takes two CLI arguments: `<texts.json> <output_batch.jsonl>`. Uses a placeholder `MODEL_NAME = "your_model"` that must be set to your fine-tuned model name.
  - `create_batch_upload.py`: uploads the batch request file and creates the OpenAI batch job (`/v1/chat/completions`, `completion_window="24h"`).
  - `process_monitor.py`: checks a batch job's status and progress (requires `JOB_ID`).
  - `download_batch_file.py`: downloads batch output by `file_id` and writes it to a local file.

- `code_for_finetuning/`: start fine-tuning and retrieve results.
  - `uploader_for_finetuning.py`: uploads training + validation JSONL files and creates a fine-tuning job. Currently hardcodes `train.jsonl` and `validation.jsonl` as input files, and uses `gpt-4o-mini-2024-07-18` as the base model. Update these values to match your setup.
  - `process_monitor.py`: checks fine-tuning job status (requires `job_id`).
  - `download_model_name.py`: fetches the fine-tuned model name (requires `job_id`).

- `datasets/`: dataset JSON files used as sources/ground truth.
  - `gt_text_webnlg*.json`: ground-truth WebNLG variants.
  - `gt_text_hotmusvalid.json`: ground-truth HotMusValid.
  - `remove_row.py`: dataset cleanup stub (uses placeholder filenames).

- `evaluation/`: evaluation utilities (triple parsing + fuzzy matching metrics).
  - `evaluation.py`: compares predicted triples vs ground truth triples, using:
    - DBpedia Spotlight for entity linking
    - DBpedia SPARQL `owl:sameAs` for sameAs equivalence
    - `sentence-transformers` (`all-MiniLM-L6-v2`) cosine similarity for predicate matching
    - Writes a per-document JSON report and computes precision/recall/F1
    - Uses placeholder filenames in `__main__` (`example_groundtruth.json`, `example_predictions.json`) — update before running.
  - `formater_for_evaluation.py`: parses OpenAI batch output `.jsonl` into a JSON structure with `custom_id` and extracted triple strings. Takes two CLI arguments via `argparse`: `<input_file> <output_file>`.
  - `duplicate_cleaner.py`: removes duplicates from each item's `content` list (expects `content` to be a list).

- `inference_test/`: quick smoke-test for prompting/inference.
  - `run_inference.py`: calls `client.chat.completions.create(...)` using a fine-tuned gpt-4o-mini model for a hardcoded `texts` list, then de-duplicates and prints "cleaned" Turtle lines. Update the model name and input texts before use.
  - `run_inference_yaml.py`: runs inference using an `inference.yaml` file (pass with `--config`).

- `prompts/`: prompt text fragments used for few-shot / prompting variants.
  - `p0.txt` … `p4.txt`: prompt templates.

- `train_validation_sets/`: converts dataset compilation JSON into fine-tuning JSONL.
  - `set_creator/set_creator.py`: reads `train_compilation.json` and `validation_compilation.json` (filenames are configurable near the top) and writes:
    - `train_set.jsonl`
    - `validation_set.jsonl`

## OpenAI Configuration Notes

Most scripts currently use placeholders like `OpenAI(api_key="your_key")` and hardcoded filenames/model names. Before running, you must update:
- Your API key in each script
- Any placeholder `job_id`, `file_id`, and input/output filenames
- `MODEL_NAME` in `batch_test/batch_creator.py` (set to `"your_model"`)
- Training/validation filenames in `code_for_finetuning/uploader_for_finetuning.py` (currently `train5.jsonl` / `validation5.jsonl`)
- Ground truth and prediction filenames in `evaluation/evaluation.py` (currently `example_groundtruth.json` / `example_predictions.json`)
This project uses a central `config.yaml` file to manage API keys, model names, and generation parameters to avoid hardcoding sensitive information in the scripts. 

**Before running any scripts, you must:**
1. Create a `config.yaml` file in the root directory (make sure this is added to your `.gitignore`!).
2. Define your parameters. 

**Example `config.yaml` structure:**
```yaml
api_key: "sk-your-openai-api-key"
model: "ft:gpt-4o-mini-2024-07-18:personal::DEtnoKmS" # Our fine-tuned model name
temperature: 0.0

# Base instructions for the model
system_prompt: |
  # Role: ... [Insert full prompt here]

# For quick inference tests
input_texts:
  - "Example text 1..."
  - "Example text 2..."
```

## Expected Data Schemas

### Fine-tuning JSONL (`train_set.jsonl`, `validation_set.jsonl`)
Each line is a chat-example with `messages` including:
- `system`: RDF triple generation instruction (DBpedia/Turtle aligned)
- `user`: the input text
- `assistant`: the target Turtle triples as plain content

Generated by: `train_validation_sets/set_creator/set_creator.py`

### Batch request `.jsonl` (for inference)
Built by: `batch_test/batch_creator.py`
- Input JSON entries should have:
  - `row_idx` (used as `custom_id`)
  - `text` (the raw input text)
- Output JSONL lines include:
  - `custom_id`
  - `method`: `POST`
  - `url`: `/v1/chat/completions`
  - `body.model`: the chat model name (set in the script via `MODEL_NAME`)
  - `body.messages`: system + user prompt(s)

### Evaluation inputs for `evaluation.py`
`evaluation.py` expects two JSON files:
- Ground truth JSON: items with `row_idx` and `triples` fields
- Predictions JSON: items with `custom_id` and `content` (list of triple strings)

The `formater_for_evaluation.py` script produces the predictions JSON shape expected by `evaluation.py`.

## Typical End-to-End Workflow

1. **Create fine-tuning JSONL**
   - Put your `train_compilation.json` and `validation_compilation.json` where `set_creator.py` expects them.
   - Run: `python train_validation_sets/set_creator/set_creator.py`

2. **Upload + start a fine-tuning job**
   - Update filenames and API key in the script, then run:
   - `python code_for_finetuning/uploader_for_finetuning.py`

3. **Monitor + fetch the fine-tuned model name**
   - `python code_for_finetuning/process_monitor.py`
   - `python code_for_finetuning/download_model_name.py`

4. **Prepare batch inference requests**
   - Set `MODEL_NAME` to your fine-tuned model name in `batch_creator.py`, then run:
   - `python batch_test/batch_creator.py <texts.json> <output_batch.jsonl>`

5. **Submit batch inference job**
   - Edit the input filename passed to `submit_batch(...)`, then run:
   - `python batch_test/create_batch_upload.py`

6. **Monitor + download batch results**
   - `python batch_test/process_monitor.py <JOB_ID>`
   - `python batch_test/download_batch_file.py <OUTPUT_FILE_ID> <OUTPUT_FILENAME>`

7. **Format + evaluate**
   - `python evaluation/formater_for_evaluation.py <input_batch_output.jsonl> <output_predictions.json>`
   - `python evaluation/duplicate_cleaner.py <output_predictions.json>` (removes duplicates from each item's `content` list)
   - Update placeholder filenames in `evaluation.py`, then run:
   - `python evaluation/evaluation.py`

## Dependencies (Inferred from Imports)

- `openai`
- `requests`
- `scikit-learn` (precision/recall/F1)
- `SPARQLWrapper`
- `sentence-transformers`
- `tqdm`
- `pyyaml`