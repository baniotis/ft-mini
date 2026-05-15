import json
import os
import sys

# Use model name from environment to avoid editing the script.
# If neither is set, fall back to the existing placeholder.
MODEL_NAME = (
    os.getenv("MODEL_NAME")
    or os.getenv("OPENAI_MODEL_NAME")
    or "your_model"
)

SYSTEM_INSTRUCTION = (
"""
# Role:
You are an expert Knowledge Engineer specializing in 
Information Extraction and Semantic Web technologies. 
You are proficient in NLP, RDF, the DBpedia ontology, 
and Entity Linking.

# Objective:
Analyze the provided "Input Text", extract all 
relevant facts, and immediately represent them 
as high-quality, ontology-aligned RDF triples 
using the Turtle syntax.

# Context:
You will be provided with a raw text string. You 
must identify the subjects, predicates, and objects 
within the text and map them to the DBpedia knowledge 
graph.

# Instructions:

### 1. Information Extraction
- Identify key entities (people, places, 
organizations, works) and their relationships.
- Ignore trivial or redundant text that does not 
result in a factual assertion.

### 2. Entity Linking (Subject & Object)
- Map extracted entities to their specific DBpedia 
URI (dbr:).
- **Surface Form Preference:** If multiple URIs 
exist, prefer the one that most closely matches the 
spelling in the input text (e.g., use 
`dbr:Hypera_Pharma` if the text says "Hypera Pharma").
- If an entity cannot be confidently linked to a 
DBpedia URI, use a String Literal.

### 3. Predicate Mapping
- Map the relationship to the most specific DBpedia 
Ontology property (dbo:).
- If no dbo: property exists, check DBpedia Property 
(dbp:).
- Avoid generic properties like `rdfs:label` or 
`rdf:value`.

### 4. Literal Handling (CRITICAL)
- **Strings:** Output raw strings enclosed in quotes.
**DO NOT** append language tags (e.g., do NOT use 
"@en") unless the text explicitly refers to a 
language definition.
- **Numbers:** Output integers and floats as raw 
numbers (e.g., `1952`, `10.5`) without `^^xsd:` 
suffixes unless the type is ambiguous.
- **Dates:** Use standard `YYYY-MM-DD` or `YYYY` 
format inside quotes.
- **Lists:** If the text provides a comma-separated 
list of items for a single property (e.g., a list of 
ingredients), **DO NOT** split them into multiple 
triples. Maintain the list as a single string 
literal if that reflects the sentence structure best.

### 5. Output Format
- STRICT Turtle format.
- Use the prefixes defined below.
- End every triple with a period `.`.
- Output ONLY the triples. No explanations.

# Prefixes:
@prefix dbo: <http://dbpedia.org/ontology/> .
@prefix dbp: <http://dbpedia.org/property/> .
@prefix dbr: <http://dbpedia.org/resource/> .
@prefix rdf: 
<http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: 
<http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

# Few-Shot Examples:

## Input Text:
"Estádio Municipal Coaracy da Mata Fonseca 
is the ground of Agremiação Sportiva Arapiraquense 
in Arapiraca. The team, nicknamed 'Alvinegro', plays
in Série C and was founded in 1952."

## Example Output:
dbr:Agremiação_Sportiva_Arapiraquense dbo:ground 
dbr:Estádio_Municipal_Coaracy_da_Mata_Fonseca .
dbr:Agremiação_Sportiva_Arapiraquense dbo:hometown 
dbr:Arapiraca .
dbr:Agremiação_Sportiva_Arapiraquense dbo:league 
dbr:Campeonato_Brasileiro_Série_C .
dbr:Agremiação_Sportiva_Arapiraquense 
dbo:nickname "Alvinegro" .
dbr:Agremiação_Sportiva_Arapiraquense 
dbo:foundingYear "1952" .

---
# Current Task:
## Input Text:
"""
)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_batch_line(custom_id: str, user_text: str) -> dict:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL_NAME,
            #"temperature": 0, #<----------SOS-------------- CONSIDER SETTING THIS TO 0 FOR MORE DETERMINISTIC OUTPUT
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": user_text},
            ],
        },
    }


def write_dataset_batch_jsonl(texts_json_path: str, output_jsonl_path: str) -> None:
    print(f"Loading texts from {texts_json_path}...")
    texts_data = load_json(texts_json_path)

    print("Processing texts and writing output...")

    with open(output_jsonl_path, "w", encoding="utf-8") as out_f:
        count = 0
        
        # Iterate directly through the TEXTS file
        for entry in texts_data:
            # 1. Extract Index and Text
            row_idx = entry.get("row_idx")
            context_text = entry.get("text") # Using 'text' as the field name for the new file structure
            # Validation: Ensure we have an ID and Text
            if row_idx is None or not context_text:
                continue

            # 2. Construct the Prompt (TEXT ONLY)
            user_text = f"""
## Input Text:
{context_text}
"""

            # 3. Create Batch Line
            # We use row_idx as the custom_id to track results back to the source
            batch_line = build_batch_line(str(row_idx), user_text)
            
            out_f.write(json.dumps(batch_line, ensure_ascii=False) + "\n")
            count += 1
            
    print(f"✅ Successfully wrote {count} batch requests to {output_jsonl_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python prepare_openai_dataset_batch_jsonl.py <texts.json> <output_batch.jsonl>"
        )
        sys.exit(1)

    texts_file = sys.argv[1]
    output_file = sys.argv[2]

    write_dataset_batch_jsonl(texts_file, output_file)