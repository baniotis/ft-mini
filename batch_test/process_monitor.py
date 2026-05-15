import argparse
import os
from openai import OpenAI
from typing import Optional


def _get_api_key(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    raise ValueError(
        "Missing OpenAI API key. Provide --api-key or set environment variable OPENAI_API_KEY."
    )

def check_status(job_id: str, api_key: str):
    client = OpenAI(api_key=api_key)
    job = client.batches.retrieve(job_id)

    print("Job ID:", job.id)
    print("   Status:", job.status)

    # request_counts is an object with attributes
    if job.request_counts:
        counts = job.request_counts
        print(f"   Progress: {counts.completed} / {counts.total} done, {counts.failed} failed")

    if job.status == "completed":
        print("    Output file ID:", job.output_file_id)
        if job.output_file_id:
            print(f"OUTPUT_FILE_ID={job.output_file_id}")
    elif job.status == "failed":
        print("    Batch failed")

    return job

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check status of an OpenAI batch job.")
    parser.add_argument("job_id", help="Batch job ID.")
    parser.add_argument("--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY env var).")
    args = parser.parse_args()

    api_key = _get_api_key(args.api_key)
    check_status(args.job_id, api_key=api_key)
