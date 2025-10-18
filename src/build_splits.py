from pathlib import Path
import sys
import csv

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

BASE_DIR = Path(r"C:\Users\MerveOzdemir\Desktop\VCTS")
SPLIT_DIR = BASE_DIR / "splits" / "v1"
META_CSV = SPLIT_DIR / "meta.csv"

TRAIN_CSV = SPLIT_DIR / "train.csv"
VAL_CSV   = SPLIT_DIR / "val.csv"
TEST_CSV  = SPLIT_DIR / "test.csv"

RANDOM_STATE = 42

REQUIRED_COLS = ["rel_path", "label", "source", "patient_id", "site"]

def main():
    if not META_CSV.exists():
        print(f"[HATA] meta.csv bulunamadı: {META_CSV}")
        sys.exit(1)

    df = pd.read_csv(META_CSV)
    for c in REQUIRED_COLS:
        if c not in df.columns:
            print(f"[HATA] meta.csv '{c}' kolonunu içermiyor.")
            sys.exit(1)

    df = df[REQUIRED_COLS].copy()

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
    y = df["label"]
    idx_trainval, idx_test = next(sss1.split(df, y))
    df_trainval = df.iloc[idx_trainval].reset_index(drop=True)
    df_test     = df.iloc[idx_test].reset_index(drop=True)

    val_ratio_within_trainval = 0.15 / 0.85

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio_within_trainval, random_state=RANDOM_STATE)
    y_tv = df_trainval["label"]
    idx_train, idx_val = next(sss2.split(df_trainval, y_tv))
    df_train = df_trainval.iloc[idx_train].reset_index(drop=True)
    df_val   = df_trainval.iloc[idx_val].reset_index(drop=True)

    df_train_out = df_train.copy()
    df_train_out["split"] = "train"

    df_val_out = df_val.copy()
    df_val_out["split"] = "val"

    df_test_out = df_test.copy()
    df_test_out["split"] = "test"

    df_train_out.to_csv(TRAIN_CSV, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")
    df_val_out.to_csv(VAL_CSV, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")
    df_test_out.to_csv(TEST_CSV, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")

    def counts(df_part, name):
        total = len(df_part)
        benign = (df_part["label"] == "benign").sum()
        malign = (df_part["label"] == "malign").sum()
        print(f"{name:6s} | total: {total:5d} | benign: {benign:5d} | malign: {malign:5d}")

    print("\n=== Bölme Özeti ===")
    counts(df_train_out, "train")
    counts(df_val_out,   "val")
    counts(df_test_out,  "test")

    print("\nKaynak (source) kırılımı:")
    for name, part in [("train", df_train_out), ("val", df_val_out), ("test", df_test_out)]:
        src_counts = part["source"].value_counts().to_dict()
        print(f" - {name}: {src_counts}")

    print(f"\nYazıldı:\n - {TRAIN_CSV}\n - {VAL_CSV}\n - {TEST_CSV}")

if __name__ == "__main__":
    main()
