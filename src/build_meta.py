from pathlib import Path
import csv
import sys

BASE_DIR = Path(r"C:\Users\MerveOzdemir\Desktop\VCTS")
RAW_DIR = BASE_DIR / "data" / "raw"
SOURCES = ["dataset", "dataset_1", "oral_cancer", "oral_cancer_1"]
CLASS_NAMES = ["benign", "malign"]
OUT_DIR = BASE_DIR / "splits" / "v1"
OUT_CSV = OUT_DIR / "meta.csv"

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def main():
    if not BASE_DIR.exists():
        print(f"[HATA] BASE_DIR bulunamadı: {BASE_DIR}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    counts = {src: {cls: 0 for cls in CLASS_NAMES} for src in SOURCES}
    missing_dirs = []

    for src in SOURCES:
        for cls in CLASS_NAMES:
            cls_dir = RAW_DIR / src / cls
            if not cls_dir.exists():
                missing_dirs.append(cls_dir)
                continue

            for f in cls_dir.rglob("*"):
                if f.is_file() and is_image(f):
                    rel_path = f.relative_to(BASE_DIR).as_posix()
                    rows.append([rel_path, cls, src, "", ""])
                    counts[src][cls] += 1

    rows.sort(key=lambda r: (r[2], r[1], r[0]))

    with OUT_CSV.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["rel_path", "label", "source", "patient_id", "site"])
        writer.writerows(rows)

    total = sum(sum(counts[src].values()) for src in SOURCES)
    print("\n=== Özet ===")
    for src in SOURCES:
        b = counts[src]["benign"]
        m = counts[src]["malign"]
        if b == 0 and m == 0:
            continue
        print(f"{src:15s} | benign: {b:5d} | malign: {m:5d} | toplam: {b+m:5d}")

    print(f"\nToplam görsel: {total}")
    print(f"Yazılan CSV  : {OUT_CSV}")

    if missing_dirs:
        print("\n(Not) Bulunamadığı için atlanan klasörler:")
        for d in missing_dirs:
            print(f" - {d}")

if __name__ == "__main__":
    main()
