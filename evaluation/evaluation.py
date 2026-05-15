import json
import re
import requests
from sklearn.metrics import precision_score, recall_score, f1_score
from SPARQLWrapper import SPARQLWrapper, JSON as SPARQL_JSON
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# === DBpedia & BERT setup ===
SPOTLIGHT_API = "https://api.dbpedia-spotlight.org/en/annotate"
CONFIDENCE = 0.5
HEADERS = {"Accept": "application/json"}

print("Loading BERT model...")
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

entity_link_cache = {}
sameas_cache = {}

# === Utils ===
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_triple(triple: str):
    if not triple:
        return None
    triple = triple.strip()

    if "|" in triple:
        parts = [p.strip() for p in triple.split("|")]
    else:
        parts = [p.strip() for p in triple.split(",")]

    if len(parts) < 3:
        return None

    subj = parts[0]
    pred = parts[1]
    obj = " ".join(parts[2:])

    # 1. Strip language tags (@en, @fr) and xsd types (^^xsd:integer)
    # This regex handles: "Text"@en OR "Text"^^xsd:string
    obj = re.sub(r'^"?([^"\\]+)"?(?:@\w{2,}|(?:\^\^xsd:\w+))?$', r'\1', obj)

    # 2. Basic cleanup
    obj = obj.replace("\\", "").replace("''", "").strip(' "\'')

    # 3. Normalize numbers (1604.0 -> 1604)
    # If the object looks like a float ending in .0, strip it
    if re.match(r'^-?\d+\.0$', obj):
        obj = obj[:-2]

    subj = re.sub(r"^\w+:", "", subj)
    pred = re.sub(r"^\w+:", "", pred)
    obj = re.sub(r"^\w+:", "", obj)

    return f"{subj} | {pred} | {obj}"

# === Entity linking & sameAs ===
def link_entity(entity):
    entity = entity.replace("_", " ").strip()
    if entity in entity_link_cache:
        return entity_link_cache[entity]

    try:
        response = requests.get(
            SPOTLIGHT_API,
            headers=HEADERS,
            params={"text": entity, "confidence": CONFIDENCE},
            timeout=5
        )
        if response.ok:
            resources = response.json().get("Resources", [])
            if resources:
                uri = resources[0]["@URI"]
                entity_link_cache[entity] = uri
                return uri
    except Exception:
        pass

    entity_link_cache[entity] = None
    return None

def get_sameas_uris(uri):
    if uri in sameas_cache:
        return sameas_cache[uri]

    sameas_set = set()
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setQuery(f"""
        SELECT ?same WHERE {{
            <{uri}> <http://www.w3.org/2002/07/owl#sameAs> ?same
        }}
    """)
    sparql.setReturnFormat(SPARQL_JSON)

    try:
        results = sparql.query().convert()
        for r in results["results"]["bindings"]:
            sameas_set.add(r["same"]["value"])
    except Exception:
        pass

    sameas_cache[uri] = sameas_set
    return sameas_set

def same_entity(e1, e2):
    if e1.lower() == e2.lower():
        return "exact"

    uri1 = link_entity(e1)
    uri2 = link_entity(e2)

    if not uri1 or not uri2:
        return None

    if uri1 == uri2:
        return "linked"

    if uri2 in get_sameas_uris(uri1) or uri1 in get_sameas_uris(uri2):
        return "sameAs"

    return None

# === Predicate similarity ===
def predicate_similar_bert(p1, p2, threshold=0.5):
    if p1.lower() == p2.lower():
        return True, 1.0

    emb1 = bert_model.encode(p1, convert_to_tensor=True)
    emb2 = bert_model.encode(p2, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).item()
    return score >= threshold, score

# === Soft matching with explanation ===
def soft_match(trip1, trip2):
    try:
        s1, p1, o1 = [x.strip() for x in trip1.split("|")]
        s2, p2, o2 = [x.strip() for x in trip2.split("|")]
    except ValueError:
        return None

    subj_match = same_entity(s1, s2)
    obj_match = same_entity(o1, o2)

    pred_ok, pred_score = predicate_similar_bert(p1, p2)
    po_ok, po_score = predicate_similar_bert(f"{p1} {o1}", f"{p2} {o2}", 0.75)

    if subj_match and obj_match and (pred_ok or po_ok):
        return {
            "match_type": "soft",
            "subject_match": subj_match,
            "object_match": obj_match,
            "predicate_similarity": pred_score,
            "predicate_object_similarity": po_score
        }

    if subj_match and p1 == p2 and po_ok:
        return {
            "match_type": "soft_object",
            "subject_match": subj_match,
            "object_match": obj_match,
            "predicate_object_similarity": po_score
        }

    return None

# === Evaluation ===
def evaluate_triples(ground_truth, predictions, report_path="evaluation_report.json"):
    y_true, y_pred = [], []
    report = []

    all_ids = {doc["id"] for doc in ground_truth}

    for doc_id in tqdm(all_ids, desc="Evaluating Docs", unit="doc"):
        gt_doc = next((g for g in ground_truth if g["id"] == doc_id), None)
        pred_doc = next((p for p in predictions if p["id"] == doc_id), None)

        gt_triples = set()
        if gt_doc:
            for t in gt_doc["triples"]:
                norm = normalize_triple(t["triple"])
                if norm:
                    gt_triples.add(norm)

        pred_triples = set()
        if pred_doc:
            for t in pred_doc["output"]:
                norm = normalize_triple(t)
                if norm:
                    pred_triples.add(norm)

        matched_gt = set()
        matched_pred = set()

        doc_report = {
            "doc_id": doc_id,
            "exact_matches": [],
            "soft_matches": [],
            "false_positives": [],
            "false_negatives": []
        }

        # Exact matches
        for pt in pred_triples:
            if pt in gt_triples:
                matched_pred.add(pt)
                matched_gt.add(pt)
                doc_report["exact_matches"].append({
                    "prediction": pt,
                    "ground_truth": pt
                })

        # Soft matches
        for pt in pred_triples - matched_pred:
            for gt in gt_triples - matched_gt:
                info = soft_match(pt, gt)
                if info:
                    matched_pred.add(pt)
                    matched_gt.add(gt)
                    doc_report["soft_matches"].append({
                        "prediction": pt,
                        "ground_truth": gt,
                        **info
                    })
                    break

        # Metrics bookkeeping
        for _ in matched_gt:
            y_true.append(1)
            y_pred.append(1)

        for gt in gt_triples - matched_gt:
            y_true.append(1)
            y_pred.append(0)
            doc_report["false_negatives"].append(gt)

        for pt in pred_triples - matched_pred:
            y_true.append(0)
            y_pred.append(1)
            doc_report["false_positives"].append(pt)

        report.append(doc_report)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return precision, recall, f1

# === Main ===
if __name__ == "__main__":
    ground_truth_file = "example_groundtruth.json"
    predictions_file = "example_predictions.json"
    output_report_file = "your_evaluation_report_name.json"

    print("Loading JSON files...")
    ground_truth = load_json(ground_truth_file)
    predictions = load_json(predictions_file)

    gt_docs = [{
        "id": int(item["row_idx"]),
        "triples": [{"triple": t} for t in item["triples"]]
    } for item in ground_truth]

    pred_docs = [{
        "id": int(item["custom_id"]),
        "output": item["content"]
    } for item in predictions]

    print(f"Loaded {len(gt_docs)} GT docs and {len(pred_docs)} prediction docs.")

    precision, recall, f1 = evaluate_triples(
        gt_docs,
        pred_docs,
        report_path=output_report_file 
    )

    print("\n" + "-" * 30)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Detailed report written to {output_report_file}") 
    print("-" * 30)