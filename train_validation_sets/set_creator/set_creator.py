import json

# ====== CONFIG ======

# UPDATE THESE FILENAMES TO MATCH YOUR ACTUAL FILES
INPUT_TRAIN_JSON = "train_compilation.json" 
INPUT_VAL_JSON = "validation_compilation.json"

OUTPUT_TRAIN_JSONL = "train_set.jsonl"
OUTPUT_VAL_JSONL = "validation_set.jsonl"

SYSTEM_PROMPT = """
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
<INSERT TEXT HERE>
"""

# ====== HELPERS ======

def is_number(s):
    """Checks if a string represents a number (int or float)."""
    try:
        float(s.replace(',', '')) 
        return True
    except ValueError:
        return False

def is_literal(obj):
    obj = obj.strip()
    if obj.startswith('"') and obj.endswith('"'): return True
    if " " in obj: return True
    if is_number(obj): return True
    if obj and obj[0].isdigit() and "-" in obj: return True
    return False

def format_predicate(pred):
    pred = pred.strip()
    if "/" in pred:
        pred = pred.split("/")[-1]
    return f"dbo:{pred}"

def format_object(obj):
    obj = obj.strip()
    if is_literal(obj):
        if is_number(obj) and not (obj.startswith('"') and obj.endswith('"')):
            return obj
        return obj if obj.startswith('"') else f'"{obj}"'
    return f"dbr:{obj}"

def triples_to_turtle(triples_list):
    lines = []
    for triple in triples_list:
        parts = [x.strip() for x in triple.split("|")]
        if len(parts) < 3: continue
        subj, pred, obj = parts[0], parts[1], parts[2]
        line = f"dbr:{subj} {format_predicate(pred)} {format_object(obj)} ."
        lines.append(line)
    return "\n".join(lines)

def make_example(entry):

    text = entry.get("text", "") 
    if not text:
        # Fallback if structure is different (e.g. lexicalisation vs target)
        text = entry.get("lexicalisation", "")

    triples = triples_to_turtle(entry["triples"])
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
            {"role": "assistant", "content": triples}
        ]
    }

def process_file(input_path, output_path):
    """Reads a JSON file, converts it, and writes a JSONL file."""
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {input_path}.")
        return 0

    converted_data = [make_example(e) for e in data]

    with open(output_path, "w", encoding="utf-8") as f:
        for obj in converted_data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            
    return len(converted_data)

# ====== MAIN EXECUTION ======

if __name__ == "__main__":
    print(f"Processing Train Data from {INPUT_TRAIN_JSON}...")
    train_count = process_file(INPUT_TRAIN_JSON, OUTPUT_TRAIN_JSONL)
    
    print(f"Processing Validation Data from {INPUT_VAL_JSON}...")
    val_count = process_file(INPUT_VAL_JSON, OUTPUT_VAL_JSONL)

    print("-" * 30)
    print("Done!")
    print(f"Train examples generated: {train_count}")
    print(f"Validation examples generated: {val_count}")