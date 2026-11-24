import pandas as pd
import config

total, pos, neg = 0, 0, 0
missing = None
cols = None

for chunk in pd.read_csv(config.RAW_CSV, chunksize=config.CHUNKSIZE):
    cols = list(chunk.columns)
    total += len(chunk)
    p = int(chunk[config.LABEL].sum())
    pos += p
    neg += len(chunk) - p
    m = chunk.isna().sum()
    missing = m if missing is None else missing + m
    print(f"rows so far: {total:,}", flush=True)

print("\nColumns:", cols)
print(f"\nTotal rows: {total:,}")
print(f"{config.LABEL}=True: {pos:,} | False: {neg:,} | rate: {pos/total:.8%}")
print("\nMissing values per column:\n", missing)
