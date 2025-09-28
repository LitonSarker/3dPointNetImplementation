"""
make_splits.py
Recursively scans src_dir/Area_*/**/*.ply and writes train.txt / eval.txt.

Usage examples:
  # Area-based split (recommended for S3DIS)
  python make_splits.py --src_dir "D:\\3d-point-cloud\\out_ply\\S3DIS\\out_ply" --eval_area Area_5

  # Random 80/20
  python make_splits.py --src_dir "D:\\3d-point-cloud\\out_ply\\S3DIS\\out_ply" --val_ratio 0.2
"""
import os
import argparse
from pathlib import Path
import random

def find_plys(src_dir: Path):
    # matches Area_1..Area_6 subtrees
    return [p.resolve() for p in src_dir.glob("Area_*/*/*.ply")]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True,
                    help="Root containing Area_1..Area_6 subfolders with .ply")
    ap.add_argument("--out_dir", default="",
                    help="Where to write train.txt/eval.txt (default = src_dir)")
    ap.add_argument("--eval_area", default="",
                    help="If set (e.g., Area_5), use that entire Area for eval")
    ap.add_argument("--val_ratio", type=float, default=0.2,
                    help="Used ONLY if --eval_area not set")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src_dir)
    out_root = Path(args.out_dir) if args.out_dir else src
    out_root.mkdir(parents=True, exist_ok=True)

    ply_files = find_plys(src)
    if not ply_files:
        print(f"[ERR] No .ply files found under {src}/Area_*/**/*.ply")
        return
    print(f"[INFO] Found {len(ply_files)} .ply under {src}")

    train_list, eval_list = [], []

    if args.eval_area:
        target = args.eval_area.lower()
        for p in ply_files:
            area = p.parents[2].name  # .../Area_X/room/file.ply
            (eval_list if target == area.lower() else train_list).append(str(p))
        if not eval_list:
            print(f"[WARN] No files matched eval area '{args.eval_area}', falling back to random split.")
    if not args.eval_area or not eval_list:
        rnd = random.Random(args.seed)
        rnd.shuffle(ply_files)
        n_val = max(1, int(len(ply_files) * args.val_ratio))
        eval_list = list(map(str, ply_files[:n_val]))
        train_list = list(map(str, ply_files[n_val:]))

    # Write lists
    train_txt = out_root / "train.txt"
    eval_txt  = out_root / "val.txt"
    train_txt.write_text("\n".join(train_list), encoding="utf-8")
    eval_txt.write_text("\n".join(eval_list),  encoding="utf-8")

    # Small stats per Area
    def area_counts(paths):
        c = {}
        for s in paths:
            area = Path(s).parents[2].name
            c[area] = c.get(area, 0) + 1
        return ", ".join(f"{k}:{v}" for k,v in sorted(c.items()))
    print(f"[OK] Wrote {len(train_list)} train → {train_txt}  ({area_counts(train_list)})")
    print(f"[OK] Wrote {len(eval_list)}  eval → {eval_txt}   ({area_counts(eval_list)})")

if __name__ == "__main__":
    main()
