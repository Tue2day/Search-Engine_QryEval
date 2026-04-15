import ast
import csv
import os


OUTPUT_DIR = "OUTPUT_DIR/HW5-EXP1"
LONG_CSV_PATH = f"{OUTPUT_DIR}/HW5_Exp1_Metrics.csv"

LONG_EXPERIMENTS = [
    ("HW5-Exp-1.1a", "bm25", "firstp", 50),
    ("HW5-Exp-1.1b", "bm25", "firstp", 100),
    ("HW5-Exp-1.1c", "bm25", "firstp", 150),
    ("HW5-Exp-1.1d", "bm25", "firstp", 200),
    ("HW5-Exp-1.2a", "bm25", "bestp", 50),
    ("HW5-Exp-1.2b", "bm25", "bestp", 100),
    ("HW5-Exp-1.2c", "bm25", "bestp", 150),
    ("HW5-Exp-1.2d", "bm25", "bestp", 200),
    ("HW5-Exp-1.3a", "dense", "firstp", 50),
    ("HW5-Exp-1.3b", "dense", "firstp", 100),
    ("HW5-Exp-1.3c", "dense", "firstp", 150),
    ("HW5-Exp-1.3d", "dense", "firstp", 200),
    ("HW5-Exp-1.4a", "dense", "bestp", 50),
    ("HW5-Exp-1.4b", "dense", "bestp", 100),
    ("HW5-Exp-1.4c", "dense", "bestp", 150),
    ("HW5-Exp-1.4d", "dense", "bestp", 200),
]

TABLES = [
    {
        "filename": "HW5_Exp1_Table_1.csv",
        "title": "BM25 retrieval, first, 5 passages",
        "experiments": [
            ("HW5-Exp-1.1a", "Exp-1.1a", 50),
            ("HW5-Exp-1.1b", "Exp-1.1b", 100),
            ("HW5-Exp-1.1c", "Exp-1.1c", 150),
            ("HW5-Exp-1.1d", "Exp-1.1d", 200),
        ]
    },
    {
        "filename": "HW5_Exp1_Table_2.csv",
        "title": "BM25 retrieval, bestp, 5 passages",
        "experiments": [
            ("HW5-Exp-1.2a", "Exp-1.2a", 50),
            ("HW5-Exp-1.2b", "Exp-1.2b", 100),
            ("HW5-Exp-1.2c", "Exp-1.2c", 150),
            ("HW5-Exp-1.2d", "Exp-1.2d", 200),
        ]
    },
    {
        "filename": "HW5_Exp1_Table_3.csv",
        "title": "dense retrieval, firstp, 5 passages",
        "experiments": [
            ("HW5-Exp-1.3a", "Exp-1.3a", 50),
            ("HW5-Exp-1.3b", "Exp-1.3b", 100),
            ("HW5-Exp-1.3c", "Exp-1.3c", 150),
            ("HW5-Exp-1.3d", "Exp-1.3d", 200),
        ]
    },
    {
        "filename": "HW5_Exp1_Table_4.csv",
        "title": "dense retrieval, bestp, 5 passages",
        "experiments": [
            ("HW5-Exp-1.4a", "Exp-1.4a", 50),
            ("HW5-Exp-1.4b", "Exp-1.4b", 100),
            ("HW5-Exp-1.4c", "Exp-1.4c", 150),
            ("HW5-Exp-1.4d", "Exp-1.4d", 200),
        ]
    },
]


def read_runtime(exp_id):
    log_path = f"{OUTPUT_DIR}/{exp_id}.log"
    if not os.path.exists(log_path):
        return("")

    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("Time:"):
                return(line.split("Time:", 1)[1].strip())
    return("")


def read_te_metrics(exp_id):
    metrics = {"MRR": "", "P@1": "", "P@5": ""}
    te_path = f"{OUTPUT_DIR}/{exp_id}.teOut"
    if not os.path.exists(te_path):
        return(metrics)

    with open(te_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            metric_name, _, value = parts
            if metric_name == "recip_rank":
                metrics["MRR"] = value
            elif metric_name == "P_1":
                metrics["P@1"] = value
            elif metric_name == "P_5":
                metrics["P@5"] = value
    return(metrics)


def read_qa_metrics(exp_id):
    metrics = {"exact": "", "f1": ""}
    qa_path = f"{OUTPUT_DIR}/{exp_id}.qaOut"
    if not os.path.exists(qa_path):
        return(metrics)

    last_dict = None
    with open(qa_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                last_dict = line

    if last_dict is None:
        return(metrics)

    data = ast.literal_eval(last_dict)
    metrics["exact"] = data.get("exact_match", "")
    metrics["f1"] = data.get("f1", "")
    return(metrics)


def format_exact_or_f1(value):
    if value == "":
        return("")
    return(f"{float(value):.2f}")


def format_rank_metric(value):
    if value == "":
        return("")
    return(f"{float(value):.4f}")


def collect_metrics(exp_id):
    qa = read_qa_metrics(exp_id)
    te = read_te_metrics(exp_id)
    runtime = read_runtime(exp_id)
    return({
        "exact": format_exact_or_f1(qa["exact"]),
        "f1": format_exact_or_f1(qa["f1"]),
        "MRR": format_rank_metric(te["MRR"]),
        "P@1": format_rank_metric(te["P@1"]),
        "P@5": format_rank_metric(te["P@5"]),
        "time": runtime
    })


def write_long_csv():
    with open(LONG_CSV_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "exp_id", "retrieval", "selection", "psg_len",
            "exact", "f1", "MRR", "P@1", "P@5", "time"
        ])

        for exp_id, retrieval, selection, psg_len in LONG_EXPERIMENTS:
            metrics = collect_metrics(exp_id)
            writer.writerow([
                exp_id,
                retrieval,
                selection,
                psg_len,
                metrics["exact"],
                metrics["f1"],
                metrics["MRR"],
                metrics["P@1"],
                metrics["P@5"],
                metrics["time"]
            ])


def write_report_table(table_spec):
    path = f'{OUTPUT_DIR}/{table_spec["filename"]}'
    metrics_by_exp = {
        exp_id: collect_metrics(exp_id)
        for exp_id, _, _ in table_spec["experiments"]
    }

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([table_spec["title"], "", "", "", ""])
        writer.writerow(["Metrics"] + [
            f"{psg_len}\n{label}" for _, label, psg_len in table_spec["experiments"]
        ])
        for metric_name in ["exact", "f1", "MRR", "P@1", "P@5", "time"]:
            row = [metric_name]
            for exp_id, _, _ in table_spec["experiments"]:
                row.append(metrics_by_exp[exp_id][metric_name])
            writer.writerow(row)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_long_csv()
    for table_spec in TABLES:
        write_report_table(table_spec)


if __name__ == "__main__":
    main()
