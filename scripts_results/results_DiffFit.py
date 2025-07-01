from pathlib import Path
import re
from collections import defaultdict
import csv

BASE_DIR = Path("/home/ymbahram/scratch/results/DoG")

METHOD_ROOTS = {
    "all":  "DiT_inception_all_DiffFit",
    "ours": "DiT_inception_ours_DiffFit",
}

RESULT_SUBDIR_PATTERNS = ["results", "results_cg1", "results_cg1_5", "results_dog1_5"]

# --- regexes --------------------------------------------------------------
FD_RE   = re.compile(r"^\s*fd:\s*([0-9.]+)", re.I)               # first line
NUM_RE  = re.compile(r"-([0-9]+)_")                              # …-{num}_…

def read_fd_value(txt_path: Path) -> float | None:
    try:
        return float(FD_RE.match(txt_path.open().readline()).group(1))
    except Exception:
        return None

# dataset → method → list of dicts {subdir, num, fd, metric, file}
records = defaultdict(lambda: defaultdict(list))

for dataset_dir in BASE_DIR.iterdir():
    if not dataset_dir.is_dir():
        continue
    dataset = dataset_dir.name                                    # e.g. artbench-10_processed

    for method_key, root_name in METHOD_ROOTS.items():
        root_dir = dataset_dir / root_name
        if not root_dir.exists():
            continue

        # there should be exactly one sub-directory inside (whatever its name)
        subfolders = [p for p in root_dir.iterdir() if p.is_dir()]
        if not subfolders:
            continue
        ckpt_dir = subfolders[0]                                   # take the only one

        for rsd in RESULT_SUBDIR_PATTERNS:
            res_dir = ckpt_dir / rsd
            if not res_dir.exists():
                continue

            for txt_file in res_dir.glob("fd_*.txt"):
                fd_val = read_fd_value(txt_file)
                if fd_val is None:
                    continue

                # metric = dinov2 / inception / …
                metric = txt_file.stem.split("_")[1]
                # num extracted from “…-{num}_…txt”
                num_match = NUM_RE.search(txt_file.name)
                num = num_match.group(1) if num_match else "NA"

                records[dataset][method_key].append({
                    "subdir": rsd,
                    "metric": metric,
                    "num":    num,
                    "fd":     fd_val,
                    "file":   str(txt_file.relative_to(BASE_DIR))
                })

# --------------- report ---------------------------------------------------
header = ["Dataset", "Method", "Sub-dir", "Metric", "Num", "FD", "File"]
print("{:<22} | {:<5} | {:<14} | {:<9} | {:<6} | {:>8}".format(*header[:-1]))

for dataset in sorted(records):
    for method in ("all", "ours"):
        for rec in sorted(records[dataset][method], key=lambda x: (x["subdir"], x["metric"], int(x["num"]))):
            print("{:<22} | {:<5} | {:<14} | {:<9} | {:<6} | {:>8.4f}".format(
                dataset, method, rec["subdir"], rec["metric"], rec["num"], rec["fd"])
            )

# --------------- optional: save to CSV ------------------------------------
# with open("fd_values.csv", "w", newline="") as f:
#     writer = csv.DictWriter(f, fieldnames=header)
#     writer.writeheader()
#     for ds in records:
#         for mth in records[ds]:
#             for rec in records[ds][mth]:
#                 row = {
#                     "Dataset": ds, "Method": mth,
#                     "Sub-dir": rec["subdir"], "Metric": rec["metric"],
#                     "Num": rec["num"], "FD": rec["fd"], "File": rec["file"]
#                 }
#                 writer.writerow(row)
