import json
import re
from pathlib import Path


QUERY_SRC_DIR = Path("QrySet/HW1")
EXP_DIR = Path("EXP_DIR/HW1_EXP3")

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


def write_param(exp_name, qry_filename):
    param = {
        "indexPath": INDEX_PATH,
        "queryFilePath": f"EXP_DIR/HW1_EXP3/{qry_filename}",
        "task_1:ranker": {
            "type": "BM25",
            "outputLength": "1000",
            "BM25:k_1": 1.2,
            "BM25:b": 0.75
        },
        "task_2:output": {
            "type": "trec_eval",
            "outputPath": f"OUTPUT_DIR/HW1-EXP3/{exp_name}.teIn",
            "outputLength": "1000"
        }
    }

    with open(EXP_DIR / f"{exp_name}.param", "w", encoding="utf-8") as f:
        json.dump(param, f, indent=2)


# Query Composition Functions
def field_sum(terms, field):
    return "#SUM(" + " ".join(f"{t}.{field}" for t in terms) + ")"


def WSUM(terms, weights):
    parts = []
    for field, w in weights.items():
        if w > 0:
            parts.append(f"{w} {field_sum(terms, field)}")
    return "#WSUM( " + " ".join(parts) + " )"


# Load Queries
short_qs = load_queries(SHORT_FILE)
long_qs  = load_queries(LONG_FILE)


# Weight Schemes
SCHEMES = {
    "b": {  # baseline
        "body": 0.7, "title": 0.3, "url": 0.0, "keywords": 0.0, "inlink": 0.0
    },
    "c": {  # balanced
        "body": 0.4, "title": 0.3, "url": 0.1, "keywords": 0.1, "inlink": 0.1
    },
    "d": {  # title heavy
        "body": 0.3, "title": 0.4, "url": 0.2, "keywords": 0.1, "inlink": 0.0
    },
    "e": {  # keywords heavy (failure case)
        "body": 0.1, "title": 0.1, "url": 0.25, "keywords": 0.45, "inlink": 0.1
    }
}


# Experiment Configurations
experiments = [
    # Short queries
    ("3.1b", "short", "b"),
    ("3.1c", "short", "c"),
    ("3.1d", "short", "d"),
    ("3.1e", "short", "e"),

    # Long queries
    ("3.2b", "long", "b"),
    ("3.2c", "long", "c"),
    ("3.2d", "long", "d"),
    ("3.2e", "long", "e"),
]


# Generate .qry and .param files
for exp, qtype, scheme in experiments:
    entries = []
    source = short_qs if qtype == "short" else long_qs
    weights = SCHEMES[scheme]

    for qid, terms in source:
        q = WSUM(terms, weights)
        entries.append((qid, q))

    qry_name = f"HW1-Exp-{exp}.qry"
    write_qry(EXP_DIR / qry_name, entries)

    write_param(
        exp_name=f"HW1-Exp-{exp}",
        qry_filename=qry_name
    )

print("All Experiment 3 (Multifield BM25) .qry and .param files generated.")
