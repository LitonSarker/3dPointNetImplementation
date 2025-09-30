#!/usr/bin/env python3
# s3dis_annots_to_ply_general.py
# Convert S3DIS Annotations/*.txt -> room-level ASCII PLY with xyz rgb label.
# Always processes ALL Areas/rooms. Optional lists: all.txt, or train/val by areas or ratio.
# The script traverses around the Annotations/ folder, so it works with any S3DIS-like structure.
# Usage example: Listed in the Command file

import os, argparse, random
from pathlib import Path

# 13-class S3DIS mapping, as in the original dataset
# (anything else is mapped to "clutter" class 12)

S3DIS_CLASS = {
    "ceiling":0, "floor":1, "wall":2, "beam":3, "column":4, "window":5, "door":6,
    "table":7, "chair":8, "sofa":9, "bookcase":10, "board":11, "clutter":12
}

# Parse a single annotation .txt file, return list of (x,y,z,r,g,b,label) tuples
def parse_annot_file(fpath: Path, class_id: int):
    pts = []
    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 6:  # expect xyz rgb
                continue
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            r, g, b = int(float(parts[3])), int(float(parts[4])), int(float(parts[5]))
            pts.append((x, y, z, r, g, b, class_id))
    return pts

# Write points to ASCII PLY file and puts that into each original folders
def write_ply_ascii(points, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(points)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property uchar label\nend_header\n")
        for x,y,z,r,g,b,l in points:
            f.write(f"{x} {y} {z} {r} {g} {b} {l}\n")

#Travers into all folders and gets the annotation information
# Converting the raw S3DIS dataset into .ply files based on annotation information

def room_to_ply(room_dir: Path, dst_root: Path):
    ann_dir = room_dir / "Annotations"
    if not ann_dir.is_dir():
        return None
    merged = []
    for f in sorted(ann_dir.glob("*.txt")):                     #Collecting all .txt files based on annotation information
        cname = f.stem.split("_")[0].lower()
        cid = S3DIS_CLASS.get(cname, S3DIS_CLASS["clutter"])    #Mapping the class names to class ids
        merged.extend(parse_annot_file(f, cid))
    if not merged:
        return None
    # Keep Area_X/room structure
    # root/.../Area_X/room/Annotations  ->  dst/Area_X/room/room.ply
    rel = Path(room_dir.parts[-2]) / Path(room_dir.name)  # Area_X / room
    out_dir = dst_root / rel
    out_ply = out_dir / f"{room_dir.name}.ply"
    write_ply_ascii(merged, out_ply)
    return out_ply

#Preparing the list of .ply files which would be used later on..
def build_lists(ply_paths, dst: Path, make_lists: bool, val_areas, val_ratio, seed):
    if not make_lists or not ply_paths:
        return
    # Always write all.txt
    all_list = dst / "all.txt"
    all_list.write_text("\n".join(str(p) for p in ply_paths), encoding="utf-8")
    print(f"[OK] Wrote: {all_list}")

    # Optional area-based split
    if val_areas:
        val_set = set(a.lower() for a in val_areas)
        tr, va = [], []
        for p in ply_paths:
            # membership by Area name token in path
            area_token = next((part for part in Path(p).parts if part.lower().startswith("area_")), "")
            (va if area_token.lower() in val_set else tr).append(str(p))
        (dst / "train.txt").write_text("\n".join(tr), encoding="utf-8")
        (dst / "val.txt").write_text("\n".join(va), encoding="utf-8")
        print(f"[OK] Wrote area-based splits: {dst/'train.txt'}  {dst/'val.txt'} (val_areas={val_areas})")
        return

    # Optional random split by ratio, did not use it here, rather used another script to split Train and Val ratio
    if val_ratio is not None:
        rnd = random.Random(seed)
        paths = list(map(str, ply_paths))
        rnd.shuffle(paths)
        n_val = max(1, int(len(paths) * float(val_ratio)))
        va = paths[:n_val]
        tr = paths[n_val:]
        (dst / "train.txt").write_text("\n".join(tr), encoding="utf-8")
        (dst / "val.txt").write_text("\n".join(va), encoding="utf-8")
        print(f"[OK] Wrote ratio-based splits: {dst/'train.txt'}  {dst/'val.txt'} (val_ratio={val_ratio}, seed={seed})")

# The main function to parse arguments and call other functions
# Traverses all Areas/rooms under root, writes PLYs to dst mirroring structure

def main():
    ap = argparse.ArgumentParser(description="Convert S3DIS Annotations/*.txt to room-level PLYs (xyz rgb label).")
    ap.add_argument("--root", required=True, help="Path to Stanford3dDataset_v1.2_Aligned_Version")
    ap.add_argument("--dst",  required=True, help="Output root for .ply (mirrors Areas/rooms)")
    ap.add_argument("--make-lists", action="store_true", help="Also write list files (all.txt; optionally train/val)")
    ap.add_argument("--val-areas", nargs="+", default=None, help="(Optional) Which Areas go to val list, e.g., Area_5 Area_6")
    ap.add_argument("--val-ratio", type=float, default=None, help="(Optional) Random val ratio in [0,1], e.g., 0.2")
    ap.add_argument("--seed", type=int, default=42, help="Seed for ratio split")
    args = ap.parse_args()

    root, dst = Path(args.root), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    ply_paths = []
    # Traverse all Areas and rooms
    for area_dir in sorted(root.glob("Area_*")):
        if not area_dir.is_dir():
            continue
        for room_dir in sorted(area_dir.iterdir()):
            if not room_dir.is_dir():
                continue
            out_ply = room_to_ply(room_dir, dst)
            if out_ply:
                ply_paths.append(out_ply.resolve())

    print(f"[OK] Wrote {len(ply_paths)} PLY files under {dst}")
    build_lists(ply_paths, dst, args.make_lists, args.val_areas, args.val_ratio, args.seed)

#Main function call
if __name__ == "__main__":
    main()
