# CSV'lerden Ultralytics klasör yapısı üretir: splits/v1/cls/{train,val,test}/{benign,malign}
from pathlib import Path
import pandas as pd
import shutil

BASE_DIR = Path(r"C:\Users\MerveOzdemir\Desktop\VCTS")
SPLIT_DIR = BASE_DIR / "splits" / "v1"
CLS_DIR = SPLIT_DIR / "cls"

def copy_from_csv(csv_name: str):
    df_path = SPLIT_DIR / csv_name
    if not df_path.exists():
        print(f"[HATA] {df_path} bulunamadı.")
        return

    df = pd.read_csv(df_path)
    split = csv_name.split(".")[0]  # train / val / test
    total = len(df)
    ok = 0
    missing = 0

    for _, row in df.iterrows():
        src = BASE_DIR / str(row["rel_path"])
        label = str(row["label"])
        dst_dir = CLS_DIR / split / label
        dst_dir.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            print(f"[WARN] Bulunamadı: {src}")
            missing += 1
            continue

        dst = dst_dir / src.name
        if dst.exists():
            stem, suf = dst.stem, dst.suffix
            i = 1
            while dst.exists():
                dst = dst_dir / f"{stem}__{i}{suf}"
                i += 1

        shutil.copy2(src, dst)
        ok += 1

    print(f"{csv_name}: kopyalandı={ok}, eksik={missing}, toplam={total}")

def main():
    for name in ["train.csv", "val.csv", "test.csv"]:
        copy_from_csv(name)
    print(f"\nTamamlandı. Klasör: {CLS_DIR}")

if __name__ == "__main__":
    main()
