import argparse
import os
from typing import Optional
from openai import OpenAI


def _get_api_key(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    raise ValueError(
        "Missing OpenAI API key. Provide --api-key or set environment variable OPENAI_API_KEY."
    )


def submit_batch(input_file: str, api_key: str) -> str:
    client = OpenAI(api_key=api_key)

    # Upload your JSONL requests file
    batch_file = client.files.create(
        file=open(input_file, "rb"),
        purpose="batch"
    )
    print("Uploaded file:", batch_file.id)

    # Create the batch job
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print("Batch job created:", batch_job.id)
    return batch_job.id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a batch JSONL and start a Batch job.")
    parser.add_argument("input_file", help="Path to the batch requests JSONL file.")
    parser.add_argument("--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY env var).")
    args = parser.parse_args()

    api_key = _get_api_key(args.api_key)
    job_id = submit_batch(args.input_file, api_key=api_key)
    print(f"JOB_ID={job_id}")
