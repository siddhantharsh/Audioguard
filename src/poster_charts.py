import os
from pathlib import Path
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import DATA_DIR, META_CSV, AUDIO_DIR, LABEL_MAP, ARTIFACTS, CHARTS_DIR

CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# 1) Load metadata
meta = pd.read_csv(META_CSV)

# 2) Compute durations (seconds)
durations = []
classes = []
folds = []
for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Scanning durations"):
    f = AUDIO_DIR / f"fold{int(row['fold'])}" / row["slice_file_name"]
    try:
        d = librosa.get_duration(path=str(f))
    except Exception:
        d = np.nan
    durations.append(d)
    classes.append(row["class"])
    folds.append(int(row["fold"]))

df = pd.DataFrame({
    "class_orig": classes,
    "class_merged": [LABEL_MAP.get(c, "other") for c in classes],
    "fold": folds,
    "duration": durations
}).dropna()

# 3) Save a small sample table (first 10 rows) for poster
sample_csv = ARTIFACTS / "sample_table.csv"
df.head(10).to_csv(sample_csv, index=False)

# 4) Chart 1: Histogram of durations
plt.figure(figsize=(6,4))
plt.hist(df["duration"], bins=20)
plt.title("Distribution of Clip Durations")
plt.xlabel("Seconds"); plt.ylabel("Count")
plt.tight_layout()
plt.savefig(CHARTS_DIR / "hist_duration.png", dpi=200); plt.close()

# 5) Chart 2: Bar chart of counts per original class
counts_orig = df["class_orig"].value_counts().sort_values(ascending=False)
plt.figure(figsize=(7,4))
counts_orig.plot(kind="bar")
plt.title("Counts per Original Class")
plt.xlabel("Class"); plt.ylabel("Count")
plt.tight_layout()
plt.savefig(CHARTS_DIR / "bar_counts_original.png", dpi=200); plt.close()

# 6) Chart 3: Bar chart of counts per merged class
counts_merged = df["class_merged"].value_counts().sort_values(ascending=False)
plt.figure(figsize=(6,4))
counts_merged.plot(kind="bar")
plt.title("Counts per Merged Class")
plt.xlabel("Merged Class"); plt.ylabel("Count")
plt.tight_layout()
plt.savefig(CHARTS_DIR / "bar_counts_merged.png", dpi=200); plt.close()

# 7) Chart 4: Boxplot of durations by merged class
plt.figure(figsize=(7,4))
order = counts_merged.index.tolist()
data = [df[df["class_merged"]==c]["duration"].values for c in order]
plt.boxplot(data, labels=order, showfliers=True)
plt.title("Duration by Merged Class")
plt.ylabel("Seconds")
plt.tight_layout()
plt.savefig(CHARTS_DIR / "box_duration_by_class.png", dpi=200); plt.close()

# 8) Chart 5: Line chart – counts per fold (trend-like)
fold_counts = df.groupby("fold").size()
plt.figure(figsize=(6,4))
fold_counts.plot(kind="line", marker="o")
plt.title("Number of Clips per Fold")
plt.xlabel("Fold (1–10)"); plt.ylabel("Count")
plt.tight_layout()
plt.savefig(CHARTS_DIR / "line_counts_per_fold.png", dpi=200); plt.close()

# 9) Key findings (auto-generate a few)
findings = []
dur_mean = df["duration"].mean()
dur_med = df["duration"].median()
dmin, dmax = df["duration"].min(), df["duration"].max()
top_class = counts_merged.index[0]
findings.append(f"Most clips are around {dur_med:.1f}s (mean {dur_mean:.1f}s).")
findings.append(f"Durations range from {dmin:.1f}s to {dmax:.1f}s.")
findings.append(f"Most frequent merged class: {top_class}.")
(Path(ARTIFACTS / "poster_findings.txt")).write_text("\n".join(findings))

print("Saved charts in artifacts/charts and findings in artifacts/poster_findings.txt")
print("Sample table:", sample_csv)
