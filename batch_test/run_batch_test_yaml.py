import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'pyyaml'. Install with: python -m pip install pyyaml"
        ) from e

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ensure_parent_dir(file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)


def _run(cmd, env, cwd: Path):
    print(f"Running: {' '.join(str(c) for c in cmd)}")
    res = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    if res.stdout:
        print(res.stdout)
    if res.stderr:
        print(res.stderr)
    return res.stdout or ""


def _parse_id(stdout: str, label: str) -> Optional[str]:
    """
    Parses patterns like `JOB_ID=<id>` or `OUTPUT_FILE_ID=<id>` from stdout.
    """
    m = re.search(rf"{re.escape(label)}=([A-Za-z0-9:\\-_]+)", stdout)
    if not m:
        return None
    return m.group(1)


def main():
    parser = argparse.ArgumentParser(description="Run batch_test workflow from a YAML config.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("batch_test_run.yaml")),
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    repo_root = Path(__file__).resolve().parents[1]
    cfg = _load_yaml(config_path)

    openai_api_key = cfg["openai_api_key"]
    model_name = cfg["model_name"]
    texts_json = cfg["texts_json"]
    batch_output_jsonl = cfg["batch_output_jsonl"]

    submit = cfg.get("submit", True)
    monitor = cfg.get("monitor", True)
    download = cfg.get("download", True)

    job_id = cfg.get("job_id")
    output_file_id = cfg.get("output_file_id")
    download_output_filename = cfg.get("download_output_filename", "batch_outputs/batch_results.jsonl")

    # Resolve input/output paths relative to repo root.
    texts_json = (repo_root / texts_json).resolve()
    batch_output_jsonl = (repo_root / batch_output_jsonl).resolve()
    download_output_filename = (repo_root / download_output_filename).resolve()

    _ensure_parent_dir(batch_output_jsonl)
    _ensure_parent_dir(download_output_filename)

    env = dict(os.environ)
    env["OPENAI_API_KEY"] = str(openai_api_key)
    env["MODEL_NAME"] = str(model_name)

    # 1) Create batch requests JSONL
    if not texts_json.exists():
        raise FileNotFoundError(f"texts_json not found: {texts_json}")

    print("[1/4] Creating batch request JSONL...")
    _run(
        [sys.executable, str(repo_root / "batch_test" / "batch_creator.py"), str(texts_json), str(batch_output_jsonl)],
        env=env,
        cwd=repo_root,
    )

    # 2) Submit batch job
    if submit:
        print("[2/4] Submitting batch job...")
        submit_stdout = _run(
            [
                sys.executable,
                str(repo_root / "batch_test" / "create_batch_upload.py"),
                str(batch_output_jsonl),
            ],
            env=env,
            cwd=repo_root,
        )
        parsed_job_id = _parse_id(submit_stdout, "JOB_ID")
        if not job_id and parsed_job_id:
            job_id = parsed_job_id

    if not job_id:
        raise ValueError("Missing job_id. Provide it in YAML, or enable submit and ensure JOB_ID is parsed.")

    # 3) Monitor batch job to get output_file_id
    if monitor:
        print("[3/4] Monitoring batch job...")
        monitor_stdout = _run(
            [sys.executable, str(repo_root / "batch_test" / "process_monitor.py"), str(job_id)],
            env=env,
            cwd=repo_root,
        )
        parsed_output_file_id = _parse_id(monitor_stdout, "OUTPUT_FILE_ID")
        if not output_file_id and parsed_output_file_id:
            output_file_id = parsed_output_file_id

    if not output_file_id:
        raise ValueError("Missing output_file_id. Provide it in YAML or enable monitor and ensure OUTPUT_FILE_ID is parsed.")

    # 4) Download results
    if download:
        print("[4/4] Downloading batch results...")
        _run(
            [
                sys.executable,
                str(repo_root / "batch_test" / "download_batch_file.py"),
                str(output_file_id),
                str(download_output_filename),
            ],
            env=env,
            cwd=repo_root,
        )
    else:
        print("Download step disabled (download=false).")


if __name__ == "__main__":
    main()

