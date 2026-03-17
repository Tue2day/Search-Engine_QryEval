import json
import os

# ==============================================================================
# Configuration
# ==============================================================================

# Paths
INDEX_PATH = "INPUT_DIR/index-cw09"
QUERY_FILE = "EXP_DIR/HW2_EXP1/HW2-Exp-Bow.qry" # 复用相同的查询文件

# Output Directories
PARAM_DIR = "EXP_DIR/HW2_EXP2"
OUTPUT_DIR = "OUTPUT_DIR/HW2-EXP2"

# Input from Experiment 1 (Required for re-ranking)
BASELINE_INRANK = "OUTPUT_DIR/HW2-EXP1/HW2-Exp-1.1a.teIn"

# Fixed Parameters
NUM_DOCS = "10"
NUM_TERMS = "10"
BM25_K1 = "1.2"
BM25_B = "0.75"

# Ensure directories exist
if not os.path.exists(PARAM_DIR):
    os.makedirs(PARAM_DIR)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==============================================================================
# Experiment Definitions
# ==============================================================================

# 2.1x = Okapi, 2.2x = RM3
# Suffixes:
# b: title -> title
# c: title -> body
# d: keywords -> body
# e: url -> body
# f: inlink -> body

experiments = [
    # --- Okapi Group (2.1x) ---
    {"id": "HW2-Exp-2.1b", "algo": "okapi", "in": "title",    "out": "title"},
    {"id": "HW2-Exp-2.1c", "algo": "okapi", "in": "title",    "out": "body"},
    {"id": "HW2-Exp-2.1d", "algo": "okapi", "in": "keywords", "out": "body"},
    {"id": "HW2-Exp-2.1e", "algo": "okapi", "in": "url",      "out": "body"},
    {"id": "HW2-Exp-2.1f", "algo": "okapi", "in": "inlink",   "out": "body"},

    # --- RM3 Group (2.2x) ---
    {"id": "HW2-Exp-2.2b", "algo": "rm3",   "in": "title",    "out": "title"},
    {"id": "HW2-Exp-2.2c", "algo": "rm3",   "in": "title",    "out": "body"},
    {"id": "HW2-Exp-2.2d", "algo": "rm3",   "in": "keywords", "out": "body"},
    {"id": "HW2-Exp-2.2e", "algo": "rm3",   "in": "url",      "out": "body"},
    {"id": "HW2-Exp-2.2f", "algo": "rm3",   "in": "inlink",   "out": "body"},
]

print(f"Generating parameters for {len(experiments)} experiments...")

for exp in experiments:
    param_filename = f"{PARAM_DIR}/{exp['id']}.param"
    
    config = {
        "indexPath": INDEX_PATH,
        "queryFilePath": QUERY_FILE,
        
        # Task 1: Read Exp 1 Baseline
        "task_1:ranker": {
            "type": "inRankFile",
            "inRankFile:Path": BASELINE_INRANK
        },
        
        # Task 2: Rewriter
        "task_2:rewriter": {
            "type": "prf",
            "prf:algorithm": exp['algo'],
            "prf:numDocs": NUM_DOCS,
            "prf:numTerms": NUM_TERMS,
            "prf:expansionFieldIn": exp['in'],
            "prf:expansionFieldOut": exp['out'],
            "prf:expansionQueryFile": f"{OUTPUT_DIR}/{exp['id']}.qryOut",
            "prf:rm3:origWeight": "0.0"
        },
        
        # Task 3: Ranker
        "task_3:ranker": {
            "type": "BM25",
            "BM25:k_1": BM25_K1,
            "BM25:b": BM25_B,
            "outputLength": "1000"
        },
        
        # Task 4: Output
        "task_4:output": {
            "type": "trec_eval",
            "outputPath": f"{OUTPUT_DIR}/{exp['id']}.teIn",
            "outputLength": "1000"
        }
    }
    
    with open(param_filename, 'w') as f:
        json.dump(config, f, indent=4)
        
print("Done. Parameter files saved to:", PARAM_DIR)