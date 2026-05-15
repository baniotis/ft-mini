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

def download_results(file_id: str, output_file: str, api_key: str):
    client = OpenAI(api_key=api_key)
    result_file = client.files.content(file_id)
    with open(output_file, "wb") as f:
        f.write(result_file.read())
    print("Results saved to", output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download OpenAI batch output file.")
    parser.add_argument("output_file_id", help="OpenAI batch output file ID.")
    parser.add_argument("output_filename", help="Path to write downloaded file.")
    parser.add_argument("--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY env var).")
    args = parser.parse_args()

    api_key = _get_api_key(args.api_key)
    download_results(args.output_file_id, args.output_filename, api_key=api_key)
