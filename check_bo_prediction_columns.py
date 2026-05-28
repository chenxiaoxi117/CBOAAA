import sys
import pandas as pd
from pathlib import Path

p = Path(sys.argv[1])

for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
    try:
        df = pd.read_csv(p, encoding=enc)
        print("encoding =", enc)
        break
    except Exception:
        df = None

if df is None:
    df = pd.read_csv(p)

keys = [
    "pred", "mu", "sigma", "std", "posterior", "acq",
    "score", "candidate", "rank", "surprise", "error",
    "selected", "source"
]

print("columns =", len(df.columns))
print("rows =", len(df))
print("\nMatched columns:")

matched = []
for c in df.columns:
    s = str(c).lower()
    if any(k in s for k in keys):
        nonnull = df[c].notna().sum()
        sample = df[c].dropna().head(3).tolist()
        matched.append(c)
        print(f"{c} | nonnull={nonnull} | sample={sample}")

print("\nTotal matched =", len(matched))

print("\nAll columns:")
for i, c in enumerate(df.columns):
    print(i, c)
