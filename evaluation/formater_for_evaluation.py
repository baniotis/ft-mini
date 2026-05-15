import json
import argparse
import re

def clean_term(term):
    """
    Cleans RDF terms, handling full URIs, prefixed terms, and typed literals.
    """
    # 1. Handle Full URIs: <http://.../Name> -> Name
    if term.startswith('<') and term.endswith('>'):
        # Remove brackets
        content = term[1:-1]
        # split by '/' and take the last part
        return content.split('/')[-1]

    # 2. Handle Typed Literals: "Value"^^xsd:type -> "Value"
    if '^^' in term:
        # Split at '^^' and keep the first part (the value)
        return term.split('^^')[0]

    # 3. Handle Prefixed terms: dbr:Name -> Name (fallback from previous script)
    # Removes dbr:, dbo:, dbp:, rdf:, etc.
    if ':' in term and not term.startswith('"'): 
        # Only do this if it looks like a prefix, not inside a quote
        return term.split(':', 1)[-1]

    return term

def parse_rdf_content(raw_text):
    if not raw_text:
        return []

    formatted_items = []
    lines = raw_text.split('\n')

    for line in lines:
        line = line.strip()
        
        # Skip empty lines or @prefix definitions
        if not line or line.startswith('@prefix'):
            continue

        # Remove trailing dot
        if line.endswith(' .'):
            line = line[:-2]

        # Split into Subject, Predicate, Object
        parts = line.split(maxsplit=2)

        if len(parts) == 3:
            s, p, o = parts

            # Clean all three parts
            s_clean = clean_term(s)
            p_clean = clean_term(p)
            o_clean = clean_term(o)

            # Format
            formatted_items.append(f"{s_clean} | {p_clean} | {o_clean}")

    return formatted_items

def process_file(input_path, output_path):
    output_list = []

    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    custom_id = data.get('custom_id')

                    # Extract content safely
                    try:
                        content_raw = data['response']['body']['choices'][0]['message']['content']
                    except (KeyError, TypeError):
                        content_raw = ""

                    content_list = parse_rdf_content(content_raw)

                    output_list.append({
                        "custom_id": custom_id,
                        "content": content_list
                    })

                except json.JSONDecodeError:
                    pass

        with open(output_path, 'w', encoding='utf-8') as outfile:
            json.dump(output_list, outfile, indent=2, ensure_ascii=False)
            
        print(f"Processed {len(output_list)} items. Saved to {output_path}")

    except FileNotFoundError:
        print(f"Error: File {input_path} not found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Input .jsonl file path")
    parser.add_argument("output_file", help="Output .json file path")
    args = parser.parse_args()

    process_file(args.input_file, args.output_file)