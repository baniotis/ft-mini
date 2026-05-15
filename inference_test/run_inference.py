from openai import OpenAI

client = OpenAI(api_key="your_key")

MODEL = "ft:gpt-4o-mini-2024-07-18:personal::DEtnoKmS" #Our finetuned model name

texts = [
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
INSERT YOUR INPUT TEXT HERE
"""
]

#EXAMPLE INPUT TEXTS:
#The 2016 Blancpain GT Sports Club was the second season of the SRO Group's Blancpain GT Sports Club, an auto racing series for grand tourer cars. The Blancpain GT Sports Club is a championship for Bronze and Iron drivers only. The \"Iron\" categorisation is within the Bronze category, for drivers over the age of 60. All drivers must participate with GT3-spec cars, RACB G3 cars or GTE-spec cars.


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

for i, text in enumerate(texts):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": text}
        ]
    )
    
    # 1. Get the raw output from the model
    raw_output = response.choices[0].message.content
    
    # 2. Run it through our cleaner function
    cleaned_output = clean_triples_string(raw_output)
    
    # 3. Print the cleaned result
    print(f"\n--- Cleaned Output {i+1} ---")
    print(cleaned_output)