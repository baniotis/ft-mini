import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'pyyaml'. Install with: python -m pip install pyyaml"
        ) from e

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_parent_dir(file_path: Path) -> None:
    parent = file_path.parent
    if str(parent) == ".":
        return
    parent.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Run evaluation steps from a YAML config.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("evaluation.yaml")),
        help="Path to evaluation YAML config.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg = _load_yaml(Path(args.config))

    input_batch_output_jsonl = cfg["input_batch_output_jsonl"]
    output_predictions_json = cfg["output_predictions_json"]
    output_clean_predictions_json = cfg["output_clean_predictions_json"]
    ground_truth_json = cfg["ground_truth_json"]
    output_report_json = cfg["output_report_json"]

    # Resolve paths relative to repo root for convenience.
    input_batch_output_jsonl = (repo_root / input_batch_output_jsonl).resolve()
    output_predictions_json = (repo_root / output_predictions_json).resolve()
    output_clean_predictions_json = (repo_root / output_clean_predictions_json).resolve()
    ground_truth_json = (repo_root / ground_truth_json).resolve()
    output_report_json = (repo_root / output_report_json).resolve()

    # Ensure output locations exist.
    _ensure_parent_dir(output_predictions_json)
    _ensure_parent_dir(output_clean_predictions_json)
    _ensure_parent_dir(output_report_json)

    # 1) formatter: .jsonl batch output -> predictions json
    from evaluation.formater_for_evaluation import process_file as format_batch_output

    print("[1/3] Formatting batch output -> predictions JSON...")
    format_batch_output(str(input_batch_output_jsonl), str(output_predictions_json))

    # 2) duplicate cleaner: de-dup each item's `content`
    from evaluation.duplicate_cleaner import clean_json_duplicates

    print("[2/3] Cleaning duplicate predictions...")
    clean_json_duplicates(str(output_predictions_json), str(output_clean_predictions_json))

    # 3) evaluation: compare predictions vs ground truth
    from evaluation.evaluation import evaluate_triples

    print("[3/3] Evaluating predictions...")
    ground_truth_raw = _load_json(ground_truth_json)
    predictions_raw = _load_json(output_clean_predictions_json)

    gt_docs: List[Dict[str, Any]] = [
        {"id": int(item["row_idx"]), "triples": [{"triple": t} for t in item["triples"]]}
        for item in ground_truth_raw
    ]
    pred_docs: List[Dict[str, Any]] = [
        {"id": int(item["custom_id"]), "output": item["content"]} for item in predictions_raw
    ]

    precision, recall, f1 = evaluate_triples(
        gt_docs,
        pred_docs,
        report_path=str(output_report_json),
    )

    print("-" * 30)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Detailed report written to {output_report_json}")


if __name__ == "__main__":
    main()

