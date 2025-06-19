import re
from pathlib import Path
import pandas as pd

# ----------------------------------------------------
# CONFIG — edit these three lines as needed
# ----------------------------------------------------

model = 'inception' # dinov2 inception
mode = "control_normalizing_exponential_cutofflatestart/50in1to1.062" # "control_film_actual" control_normalizing_exponential/50in1to2 control_normalizing_exponential/95in1to3 control_normalizing
ROOT_DIR = Path(f"/export/datasets/public/diffusion_datasets/tmp_weights/stanford-cars_processed/DiT_inception_ours/{mode}/")        # ← top-level directory
METRIC     = "fd"                            # e.g. "fd", "precision", "recall", …
OUT_CSV    = Path(f"./tables/control_normalizing_exponential95in1to3.csv") # where to write the table
# ----------------------------------------------------

# Regex helpers
re_root   = re.compile(r"W_MAX(?P<wmax>[0-9.]+)")
re_subdir = re.compile(r"_w_dgft(?P<dgft>[0-9.]+)$")
re_value  = re.compile(rf"^{METRIC}\s*:\s*(?P<val>[0-9.]+)")


table = {}

for root in ROOT_DIR.iterdir():
    if not root.is_dir():
        continue
    m_root = re_root.search(root.name)
    if not m_root:
        continue
    w_max = float(m_root.group("wmax"))

    # loop over results_* sub-folders
    for sub in root.glob("results_w_dgft*"):
        m_sub = re_subdir.search(sub.name)

        if not m_sub:
            continue
        dgft = float(m_sub.group("dgft"))

        # choose either Inception or DINOv2 file (here: Inception)
        txt_file = next(sub.glob(f"fd_{model}_*.txt"), None)
        if txt_file is None:
            raise FileNotFoundError(f"No metric file in {sub}")

        # extract the metric value
        with open(txt_file) as f:
            for line in f:
                m_val = re_value.match(line.strip())
                if m_val:
                    val = float(m_val.group("val"))
                    break
            else:
                raise ValueError(f"{METRIC} not found in {txt_file}")

        # insert into dict of dicts
        table.setdefault(dgft, {})[w_max] = val

# ---- build DataFrame -------------------------------------------------

table = dict(sorted(table.items()))

df = (
    pd.DataFrame(table)
             # rows → dgft, columns → w_max
      .sort_index()
      .reindex(sorted({k for d in table.values() for k in d}))
)
df.index.name = "w_max"
df.columns.name = "DGFT scale"

# save & show
df.to_csv(OUT_CSV)
print(f"Table saved to {OUT_CSV.absolute()}\n")
print(df.to_string(float_format="%.4g"))
