import argparse
from pathlib import Path
from typing import Optional

import yaml
from openai import OpenAI

def clean_triples_string(raw_text):
    """Takes a multi-line string of triples, removes duplicates, and returns a clean string."""
    lines = raw_text.strip().split('\n')
    seen_triples = set()
    unique_content = []
    
    for line in lines:
        clean_line = line.strip()
        # Skip empty lines, but keep unique triples (including prefixes)
        if clean_line and clean_line not in seen_triples:
            unique_content.append(clean_line)
            seen_triples.add(clean_line)
            
    # Join the unique lines back together with newlines
    return '\n'.join(unique_content)

def main():
    parser = argparse.ArgumentParser(description="Run inference test from a YAML config file.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("inference.yaml")),
        help="Path to YAML config (defaults to inference_test/inference.yaml).",
    )
    args = parser.parse_args()

    # 1. Load configuration from the YAML file
    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # 2. Initialize OpenAI Client using the key from YAML
    client = OpenAI(api_key=config['api_key'])
    
    # 3. Extract variables
    model_name = config['model']
    temperature = config.get('temperature', 0.0) # Default to 0.0 if missing
    system_prompt = config['system_prompt']
    texts = config.get('input_texts', [])

    if not texts:
        print("No texts found in config.yaml to process.")
        return

    # 4. Iterate through texts and make API calls
    for i, text in enumerate(texts):
        # Dynamically build the prompt by combining the system prompt with the current text
        full_prompt = f"{system_prompt}\n\n---\n# Current Task:\n## Input Text:\n{text}"
        
        print(f"Processing text {i+1}/{len(texts)}...")
        
        response = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            messages=[
                {"role": "user", "content": full_prompt}
            ]
        )
        
        raw_output = response.choices[0].message.content
        cleaned_output = clean_triples_string(raw_output)
        
        print(f"\n--- Cleaned Output {i+1} ---")
        print(cleaned_output)
        print("-" * 40 + "\n")

if __name__ == "__main__":
    main()