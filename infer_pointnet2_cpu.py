# infer_pointnet2_cpu.py
# CPU-only PointNet++ (SSG, pure PyTorch) inference script with:
# - optional k-NN label smoothing (--smooth_knn)
# - auto-extended color palette for any num_classes
# - reproducible subsampling (--seed)
# - normals support (--use_normals) to match 9-ch checkpoints (xyz+RGB+normals as features; xyz separate)

import argparse, csv, collections
import numpy as np
import torch  # type: ignore
import open3d as o3d  # type: ignore
from pointnet2_cpu import PointNet2_SSG_Seg

# ---------- Palette helpers ----------
BASE_PALETTE = np.array([
    [ 31,119,180],[255,127, 14],[ 44,160, 44],[214, 39, 40],
    [148,103,189],[140, 86, 75],[227,119,194],[127,127,127],
    [188,189, 34],[ 23,190,207],[174,199,232],[255,187,120],
    [152,223,138]
], dtype=np.uint8)

CLASS_NAMES = [
    "ceiling", "floor", "wall", "beam", "column", "window",
    "door", "table", "chair", "sofa", "bookcase", "board", "clutter"
]

def make_palette(n: int) -> np.ndarray:
    if n <= BASE_PALETTE.shape[0]:
        return BASE_PALETTE[:n].copy()
    extra = n - BASE_PALETTE.shape[0]
    hues = (np.arange(extra) / max(1, extra)).reshape(-1, 1)
    s = np.full((extra,1), 0.65, dtype=np.float32)
    v = np.full((extra,1), 0.95, dtype=np.float32)
    hsv = np.concatenate([hues, s, v], axis=1)
    rgb = hsv_to_rgb(hsv)
    extra_rgb = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    return np.vstack([BASE_PALETTE, extra_rgb])

def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[:,0], hsv[:,1], hsv[:,2]
    i = np.floor(h * 6).astype(int)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i_mod = i % 6
    r = np.select([i_mod==0, i_mod==1, i_mod==2, i_mod==3, i_mod==4, i_mod==5],
                  [v, q, p, p, t, v], default=0)
    g = np.select([i_mod==0, i_mod==1, i_mod==2, i_mod==3, i_mod==4, i_mod==5],
                  [t, v, v, q, p, p], default=0)
    b = np.select([i_mod==0, i_mod==1, i_mod==2, i_mod==3, i_mod==4, i_mod==5],
                  [p, p, t, v, v, q], default=0)
    return np.stack([r,g,b], axis=1)

# ---------- IO ----------
def ensure_normals(pcd: o3d.geometry.PointCloud, knn: int = 16) -> np.ndarray:
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
        try:
            pcd.orient_normals_consistent_tangent_plane(knn)
        except Exception:
            pass
    return np.asarray(pcd.normals).astype(np.float32)

def read_points_from_ply(path: str, use_rgb: bool = False, use_normals: bool = False):
    """
    Returns:
      xyz:   (N,3) float32  -> coordinates ONLY
      feats: (N,C) float32  -> EXTRA features ONLY (RGB and/or normals), C in {0,3,6}
    """
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise ValueError(f"Empty/unreadable point cloud: {path}")

    xyz = np.asarray(pcd.points, dtype=np.float32)  # (N,3)

    rgb = None
    if use_rgb:
        if pcd.has_colors():
            rgb = np.asarray(pcd.colors, dtype=np.float32)  # (N,3) in 0..1
            if rgb.max() > 1.0:  # safety
                rgb = rgb / 255.0
        else:
            rgb = np.zeros_like(xyz, dtype=np.float32)

    nrm = ensure_normals(pcd) if use_normals else None  # (N,3) or None

    parts = []
    if rgb is not None: parts.append(rgb)
    if nrm is not None: parts.append(nrm)
    feats = np.concatenate(parts, axis=1).astype(np.float32) if parts else None  # (N,C) or None
    return xyz, feats

def write_colored_ply(points_xyz, labels, out_path, palette):
    labels = labels.astype(np.int64).reshape(-1)
    labels_clamped = np.clip(labels, 0, palette.shape[0]-1)
    colors = palette[labels_clamped] / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float32))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    ok = o3d.io.write_point_cloud(out_path, pcd, write_ascii=True)
    if not ok:
        raise RuntimeError(f"Failed to write: {out_path}")

# ---------- kNN smoothing ----------
def smooth_labels_knn(points_xyz: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    N = points_xyz.shape[0]
    if k <= 1 or N == 0:
        return labels
    valid = labels >= 0
    if not np.any(valid):
        return labels
    pts_valid = points_xyz[valid]
    lbl_valid = labels[valid]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_valid.astype(np.float64))
    kdt = o3d.geometry.KDTreeFlann(pcd)

    smoothed = labels.copy()
    idx_map = np.where(valid)[0]
    for j in range(pts_valid.shape[0]):
        ok, nn_idx, _ = kdt.search_knn_vector_3d(pts_valid[j], k)
        if ok <= 0:
            continue
        votes = lbl_valid[np.asarray(nn_idx, dtype=int)]
        from collections import Counter
        c = Counter(votes.tolist()).most_common(1)[0][0]
        smoothed[idx_map[j]] = c
    return smoothed

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="CPU-only PointNet++ (SSG, pure PyTorch) inference")
    ap.add_argument("--ply", required=True, help="Input .ply path")
    ap.add_argument("--out_ply", required=True, help="Output colored .ply path")
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path")
    ap.add_argument("--use_rgb", action="store_true", help="Use RGB channels (xyz+rgb)")
    ap.add_argument("--use_normals", action="store_true", help="Use normals (xyz+rgb+normals)")
    ap.add_argument("--max_points", type=int, default=80000, help="Subsample to this many points for speed")
    ap.add_argument("--save_labels", action="store_true", help="Also save raw predicted labels as .npy and CSV")
    ap.add_argument("--smooth_knn", type=int, default=0, help="If >0, apply k-NN majority-vote smoothing")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for subsampling")
    args = ap.parse_args()

    PALETTE = make_palette(args.num_classes)
    rng = np.random.default_rng(args.seed)

    # Read: xyz separate, feats = RGB and/or normals ONLY
    xyz, feats = read_points_from_ply(args.ply, use_rgb=args.use_rgb, use_normals=args.use_normals)  # (N,3), (N,C) or None
    N = xyz.shape[0]
    C = 0 if feats is None else feats.shape[1]  # 0/3/6 (NOT including xyz)
    in_ch = C

    # Subsample
    if N > args.max_points:
        idx = rng.choice(N, args.max_points, replace=False)
        xyz_sub = xyz[idx]
        feats_sub = None if feats is None else feats[idx]
        inv_map = idx
    else:
        xyz_sub, feats_sub = xyz, feats
        inv_map = None

    # Tensors
    xyz_t = torch.from_numpy(xyz_sub).float().unsqueeze(0)  # (1, n, 3)
    if feats_sub is None:
        feats_t = None
    else:
        feats_t = torch.from_numpy(feats_sub).float().unsqueeze(0).permute(0, 2, 1)  # (1, C, n)

    # Model: in_channels must equal EXTRA features only
    model = PointNet2_SSG_Seg(in_channels=in_ch, num_classes=args.num_classes)
    model.eval()

    # Load checkpoint
    if args.ckpt:
        try:
            state = torch.load(args.ckpt, map_location="cpu")
            cand = None
            if isinstance(state, dict):
                for key in ("state_dict", "model_state_dict", "model"):
                    if key in state and isinstance(state[key], dict):
                        cand = state[key]; break
            new_state = cand if cand is not None else state
            cleaned = {k.replace("model.", "").replace("module.", ""): v for k, v in new_state.items()}
            missing, unexpected = model.load_state_dict(cleaned, strict=False)
            print(f"[CKPT] loaded with missing={len(missing)} unexpected={len(unexpected)}")
            if missing or unexpected:
                print("[CKPT] Note: Ensure in_channels matches RGB/normals only (0/3/6). XYZ is separate.")
        except Exception as e:
            print(f"[CKPT] failed to load ({e}); proceeding with random weights.")

    # Inference
    with torch.no_grad():
        logits = model(xyz_t, feats_t)  # (1, num_classes, n)
        pred_sub = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()  # (n,)

    # Map back
    if inv_map is not None:
        labels_out = np.full((N,), -1, dtype=np.int64)
        labels_out[inv_map] = pred_sub
        pts_out = xyz
    else:
        pts_out = xyz_sub
        labels_out = pred_sub

    # Optional smoothing
    if args.smooth_knn and args.smooth_knn > 0:
        print(f"[INFO] k-NN smoothing with k={args.smooth_knn}")
        labels_out = smooth_labels_knn(pts_out, labels_out, args.smooth_knn)

    # Write outputs
    write_colored_ply(pts_out, labels_out, args.out_ply, PALETTE)
    print(f"[OK] wrote colored predictions to: {args.out_ply}")

    if args.save_labels:
        npy_path = args.out_ply.replace(".ply", "_labels.npy")
        np.save(npy_path, labels_out)
        print(f"[OK] saved raw labels to: {npy_path}")

        csv_path = args.out_ply.replace(".ply", "_labels_xyz.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["x","y","z","label","class_name"])
            valid = labels_out >= 0
            for p, lbl in zip(pts_out[valid], labels_out[valid]):
                w.writerow([float(p[0]), float(p[1]), float(p[2]), int(lbl), CLASS_NAMES[int(lbl)] if 0 <= lbl < len(CLASS_NAMES) else "unknown"])
        print(f"[OK] saved CSV (x,y,z,label) to: {csv_path}")

if __name__ == "__main__":
    main()
