import json
import re
from pathlib import Path


QUERY_SRC_DIR = Path("QrySet/HW1")
EXP_DIR = Path("EXP_DIR/HW1_EXP2")

SHORT_FILE = QUERY_SRC_DIR / "short.txt"
LONG_FILE  = QUERY_SRC_DIR / "long.txt"

EXP_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = "INPUT_DIR/index-cw09"


def load_queries(path):
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            qid, text = line.strip().split(":", 1)
            text = re.sub(r"[^\w\s]", " ", text).lower()
            terms = text.split()
            queries.append((qid.strip(), terms))
    return queries


def write_qry(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        for qid, q in entries:
            f.write(f"{qid}:{q}\n")


def write_param(exp_name, qry_filename, model_type):
    ranker = {
        "type": model_type,
        "outputLength": "1000"
    }

    if model_type == "BM25":
        ranker["BM25:k_1"] = 1.2
        ranker["BM25:b"] = 0.75

    param = {
        "indexPath": INDEX_PATH,
        "queryFilePath": f"EXP_DIR/HW1_EXP2/{qry_filename}",
        "task_1:ranker": ranker,
        "task_2:output": {
            "type": "trec_eval",
            "outputPath": f"OUTPUT_DIR/HW1-EXP2/{exp_name}.teIn",
            "outputLength": "1000"
        }
    }

    with open(EXP_DIR / f"{exp_name}.param", "w", encoding="utf-8") as f:
        json.dump(param, f, indent=2)
        

# Query Composition Functions
def AND(terms, field=None):
    if field:
        return "#AND(" + " ".join(f"{t}.{field}" for t in terms) + ")"
    return "#AND(" + " ".join(terms) + ")"


def SUM(terms, field=None):
    if field:
        return "#SUM(" + " ".join(f"{t}.{field}" for t in terms) + ")"
    return "#SUM(" + " ".join(terms) + ")"


def SDM(terms):
    ordered = " ".join(
        f"#NEAR/1({terms[i]} {terms[i+1]})"
        for i in range(len(terms) - 1)
    )
    unordered = " ".join(
        f"#WINDOW/8({terms[i]} {terms[i+1]})"
        for i in range(len(terms) - 1)
    )

    return (
        "#WSUM( "
        "0.7 #SUM( " + " ".join(terms) + " ) "
        "0.2 #SUM( " + ordered + " ) "
        "0.1 #SUM( " + unordered + " ) "
        ")"
    )


# Load queries
short_qs = load_queries(SHORT_FILE)
long_qs  = load_queries(LONG_FILE)


# Experiment definitions
experiments = [
    # Ranked Boolean (title)
    ("2.1b", "short", "RB", "title"),
    ("2.1e", "long",  "RB", "title"),

    # BM25 Bag-of-Words
    ("2.2a", "short", "BM25", "body"),
    ("2.2b", "short", "BM25", "title"),
    ("2.2d", "long",  "BM25", "body"),
    ("2.2e", "long",  "BM25", "title"),

    # BM25 SDM
    ("2.2c", "short", "SDM", None),
    ("2.2f", "long",  "SDM", None),
]


# Generate .qry and .param files
for exp, qtype, model, field in experiments:

    entries = []
    source = short_qs if qtype == "short" else long_qs

    for qid, terms in source:
        if model == "RB":
            q = AND(terms, field)
            model_type = "RankedBoolean"
        elif model == "BM25":
            q = SUM(terms, field)
            model_type = "BM25"
        elif model == "SDM":
            q = SDM(terms)
            model_type = "BM25"

        entries.append((qid, q))

    qry_name = f"HW1-Exp-{exp}.qry"
    write_qry(EXP_DIR / qry_name, entries)

    write_param(
        exp_name=f"HW1-Exp-{exp}",
        qry_filename=qry_name,
        model_type=model_type
    )
print("All Experiment 2 .qry and .param files generated in EXP_DIR/HW1_EXP2/")
