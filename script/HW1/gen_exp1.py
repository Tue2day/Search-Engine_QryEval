import json
import re
from pathlib import Path


QUERY_SRC_DIR = Path("QrySet/HW1")
EXP_DIR = Path("EXP_DIR/HW1_EXP1")

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


def write_param(exp_name, qry_filename, ranked):
    param = {
        "indexPath": INDEX_PATH,
        "queryFilePath": f"EXP_DIR/HW1_EXP1/{qry_filename}",
        "task_1:ranker": {
            "outputLength": "1000",
            "type": "RankedBoolean" if ranked else "UnrankedBoolean"
        },
        "task_2:output": {
            "type": "trec_eval",
            "outputPath": f"OUTPUT_DIR/HW1-EXP1/{exp_name}.teIn",
            "outputLength": "1000"
        }
    }

    with open(EXP_DIR / f"{exp_name}.param", "w", encoding="utf-8") as f:
        json.dump(param, f, indent=2)


def AND(terms):
    return "#AND(" + " ".join(terms) + ")"


def NEAR(terms):
    return "#NEAR/3(" + " ".join(terms) + ")"


# Load queries
short_qs = load_queries(SHORT_FILE)
long_qs  = load_queries(LONG_FILE)

assert len(short_qs) == len(long_qs)

# Manual queries (YOU maintain these)
manual_short = {
    # 121 sit and reach test
    "121": "#WINDOW/10( sit reach test )",

    # 129 iowa food stamp program
    "129": "#AND( iowa #SYN(food stamp) program )",

    # 133 all men are created equal
    "133": "#WINDOW/12( all men created equal )",

    # 137 rock and gem shows
    "137": "#AND( #NEAR/3( rock gem ) #OR( shows shows.inlink ) )",

    # 141 va dmv registration
    "141": "#AND( #SYN(va virginia ).title #SYN( dmv \"department of motor vehicles\" ).title registration )",

    # 147 tangible personal property tax
    "147": "#WINDOW/15( tangible personal property tax )",

    # 155 last supper painting
    "155": "#AND( #NEAR/3( last supper) #OR( painting image picture) )",

    # 165 blue throated hummingbird
    "165": "#NEAR/3( blue throated hummingbird )",

    # 181 old coins
    "181": "#AND( #OR( old ancient ) coins.inlink )",

    # 189 gs pay rate
    "189": "#AND( #SYN(gs \"general schedule\" ) #NEAR/3( pay rate ) )",

    # 191 churchill downs
    "191": "#NEAR/3( churchill downs )",

    # 193 dog clean up bags
    "193": "#WINDOW/12( dog clean up bags )"
}

manual_long = {
    # 121 How is the sit and reach test used in physical rehabilitation treatment
    "121": "#AND( #NEAR/3(sit reach test) #OR( #NEAR/3( physical rehabilitation ) treatment.body ) )",

    # 129 How do I apply for Iowa's Food Assistance food stamps Program
    "129": "#AND( iowa #SYN( food stamp assistance ) #OR( apply application program assistance.keywords ) )",

    # 133 Who said all men are created equal
    "133": "#AND( #WINDOW/12( all men created equal ) #OR( jefferson said quote ) )",

    # 137 Locations and dates of rock and gem shows
    "137": "#AND( #NEAR/3( rock gem ) #OR( shows shows.inlink ) #OR( locations dates ) )",

    # 141 Where are offices of the Virginia Department of Motor Vehicles located and what are their hours of operation
    "141": "#AND( #SYN(va virginia ).title #SYN( dmv \"department of motor vehicles\" ).title #OR( offices hours location ) )",

    # 147 What is the definition of tangible personal property for tax purposes
    "147": "#AND( #NEAR/3( tangible personal property ) #OR( tax definition purposes definition.keywords ) )",

    # 155 Find a picture of the Last Supper painting by Leonardo da Vinci
    "155": "#AND( #NEAR/3( last supper ) #OR( leonardo \"da vinci\" ) #OR( painting image picture image.url ) )",

    # 165 What are the migration patterns of the blue-throated hummingbird
    "165": "#AND( #NEAR/3( blue throated hummingbird ) #OR( migration patterns ) )",

    # 181 What is a good way to begin collecting old coins
    "181": "#AND( #OR( old ancient ) #OR( coins coins.inlink ) #OR( collecting begin guide ) )",

    # 189 Find the General Schedule pay tables 
    "189": "#AND( #SYN( gs \"general schedule\" ) #OR( pay rate tables tables.url ) )",

    # 191 Get the racing schedule at Churchill Downs 
    "191": "#AND( #NEAR/3( churchill downs ) #OR( racing schedule ) )",

    # 193 Can I order dog clean up bags online
    "193": "#AND( #WINDOW/12( dog clean up bags ).body #OR( order online ) )"
}


# Experiment definitions
experiments = [
    # (exp_name, query_type, mode, ranked)
    ("1.1a", "short", "AND",  False),
    ("1.1b", "short", "NEAR", False),
    ("1.1c", "short", "MAN",  False),
    ("1.1d", "long",  "AND",  False),
    ("1.1e", "long",  "NEAR", False),
    ("1.1f", "long",  "MAN",  False),

    ("1.2a", "short", "AND",  True),
    ("1.2b", "short", "NEAR", True),
    ("1.2c", "short", "MAN",  True),
    ("1.2d", "long",  "AND",  True),
    ("1.2e", "long",  "NEAR", True),
    ("1.2f", "long",  "MAN",  True),
]


# Generate .qry and .param files
for exp, qtype, mode, ranked in experiments:

    entries = []
    source = short_qs if qtype == "short" else long_qs

    for qid, terms in source:
        if mode == "AND":
            q = AND(terms)
        elif mode == "NEAR":
            q = NEAR(terms)
        elif mode == "MAN":
            q = manual_short[qid] if qtype == "short" else manual_long[qid]
        entries.append((qid, q))

    qry_name = f"HW1-Exp-{exp}.qry"
    write_qry(EXP_DIR / qry_name, entries)

    write_param(
        exp_name=f"HW1-Exp-{exp}",
        qry_filename=qry_name,
        ranked=ranked
    )

print("All .qry and .param files have been generated in EXP_DIR/HW1_EXP1/")
