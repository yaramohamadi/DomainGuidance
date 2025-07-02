#!/usr/bin/env python3
import re
from pathlib import Path
import pandas as pd

# ─────────────────────────────────────────────────────────────────────
# CONFIG – edit if needed
# ─────────────────────────────────────────────────────────────────────
model   = "inception"                 # "inception" or "dinov2"
metric  = "fd"                        # "fd", "precision", "recall", …
root    = Path("/home/ymbahram/scratch/results/DoG/") #    /home/ens/AT74470/results/DoG
mode    = "control_normalizing_exponential_cutofflatestart"
datasets = [
    "stanford-cars_processed",
    "food-101_processed",
    "artbench-10_processed",
    "ffhq256",
    "cub-200-2011_processed",
        "caltech-101_processed",

]
out_dir = Path("./tables")            # CSVs will be saved here
out_dir.mkdir(exist_ok=True)
# ─────────────────────────────────────────────────────────────────────

# regex helpers (unchanged from before)
re_wmax = re.compile(r"W_MAX(?P<wmax>[0-9.]+)")
re_dgft = re.compile(r"_w_dgft(?P<dgft>[0-9.]+)$")
re_val  = re.compile(rf"^{metric}\s*:\s*(?P<val>[0-9.]+)")

def build_table(dataset: str) -> pd.DataFrame:
    parent_dir = root / dataset / f"DiT_{model}_ours" / mode
    if not parent_dir.is_dir():
        print(f"[WARN] {parent_dir} not found")
        return pd.DataFrame()  # Return empty DataFrame

    rows = []

    for mode_dir in parent_dir.iterdir():
        if not mode_dir.is_dir():
            continue
        mode_name = mode_dir.name
        for root_dir in mode_dir.iterdir():
            if not root_dir.is_dir():
                continue
            m_w = re_wmax.search(root_dir.name)
            if not m_w:
                continue
            w_max = float(m_w.group("wmax"))

            for sub in root_dir.glob("results_w_dgft*"):
                m_d = re_dgft.search(sub.name)
                if not m_d:
                    continue
                dgft = float(m_d.group("dgft"))

                val = float("nan")
                try:
                    txt = next(sub.glob(f"{metric}_{model}_*.txt"))
                    with open(txt) as f:
                        for line in f:
                            m_v = re_val.match(line.strip())
                            if m_v:
                                val = float(m_v.group("val"))
                                break
                except StopIteration:
                    print(f"[WARN] No metric file in {sub}")
                except Exception as e:
                    print(f"[WARN] Failed to parse metric in {sub}: {e}")

                rows.append(
                    dict(mode=mode_name, w_max=w_max, dgft=dgft, value=val)
                )

    df_long = pd.DataFrame(rows)
    if df_long.empty:
        return pd.DataFrame()
    df = (
        df_long
        .pivot_table(index=["mode", "w_max"], columns="dgft", values="value")
        .sort_index()
        .sort_index(axis=1)
    )
    df.index.names  = ["mode", "w_max"]
    df.columns.name = "DGFT scale"
    return df


print(datasets)
# ─────────────────────────────────────────────────────────────────────
# Main loop over datasets
# ─────────────────────────────────────────────────────────────────────
for ds in datasets:
    try:
        df = build_table(ds)
    except (FileNotFoundError, ValueError) as e:
        print(f"[WARN] {ds}: {e}")
        continue

    csv_path = out_dir / f"{ds}_{metric}.csv"
    df.to_csv(csv_path)

    print(f"\n=== {ds} ===")
    print(df.to_string(float_format="%.6g"))
    print(f"(saved to {csv_path})\n")