#!/usr/bin/env python3
# occlusion_eval_cpu.py
# Occlusion robustness eval for PointNet++ (CPU).
# - XYZ passed separately
# - "Features" = EXTRA channels only (RGB and/or normals)
# - Reads labels from PLY or from sidecar files when absent.

import argparse, os, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List, Tuple

# -------- Optional Open3D (for normals / PLY fallback) ----------
try:
    import open3d as o3d
except Exception:
    o3d = None

# ----------------- PLY parser (ASCII) : Similar as other scripts-----------------
def _parse_ascii_ply(ply_path: str):
    """
    Minimal ASCII PLY parser.
    Returns dict with: xyz:(N,3) float32, rgb:(N,3)|None float32 (0..1),
                       normals:(N,3)|None float32, label:(N,)|None int64
    """
    xyz, rgb, normals, label = [], [], [], []
    with open(ply_path, "r", encoding="utf-8") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Bad PLY: missing end_header")
            line = line.strip()
            header.append(line)
            if line == "end_header":
                break

        props = [l for l in header if l.startswith("property")]
        has_rgb     = any(p.split()[-1] in ("red", "green", "blue") for p in props)
        has_normals = any(p.split()[-1] in ("nx", "ny", "nz") for p in props)
        has_label   = any(p.split()[-1] == "label" for p in props)

        for line in f:
            vals = line.strip().split()
            if not vals:
                continue
            i = 0
            x, y, z = float(vals[i]), float(vals[i+1]), float(vals[i+2]); i += 3
            xyz.append([x, y, z])
            if has_normals:
                nx, ny, nz = float(vals[i]), float(vals[i+1]), float(vals[i+2]); i += 3
                normals.append([nx, ny, nz])
            if has_rgb:
                r, g, b = float(vals[i]), float(vals[i+1]), float(vals[i+2]); i += 3
                rgb.append([r/255.0, g/255.0, b/255.0])
            if has_label:
                label.append(int(float(vals[i]))); i += 1

    xyz = np.asarray(xyz, np.float32)
    rgb = np.asarray(rgb, np.float32) if len(rgb)     else None
    normals = np.asarray(normals, np.float32) if len(normals) else None
    label = np.asarray(label, np.int64) if len(label) else None
    return {"xyz": xyz, "rgb": rgb, "normals": normals, "label": label}

# -------- Sidecar ground-truth loader (Step 2) ----------
def _load_sidecar_labels(ply_path: str) -> Optional[np.ndarray]:
    """
    Try common sidecar locations for per-point labels (one int per row), or CSV with 'label' column.
    """
    base = Path(ply_path).with_suffix("")
    candidates = [
        str(base) + "_gt.txt",
        str(base) + "_labels.txt",
        str(base) + ".labels",
        str(base) + "_gt.csv",
        str(base) + "_labels.csv",
    ]
    for c in candidates:
        p = Path(c)
        if not p.exists():
            continue
        if p.suffix.lower() == ".csv":
            arr = []
            with open(p, "r", newline="", encoding="utf-8") as f:
                rdr = csv.reader(f)
                header = next(rdr, None)
                if header and ("label" in header):
                    li = header.index("label")
                    for row in rdr:
                        if not row: continue
                        arr.append(int(float(row[li])))
                else:
                    for row in rdr:
                        if not row: continue
                        arr.append(int(float(row[0])))
            return np.asarray(arr, dtype=np.int64)
        # text formats
        return np.loadtxt(p, dtype=np.int64)
    return None

# ----------------- metrics -----------------
# Similar to other scripts, but here we build confusion matrix directly

def confusion_matrix(num_classes: int, target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    mask = (target >= 0) & (target < num_classes)
    t = target[mask].astype(np.int64)
    p = pred[mask].astype(np.int64)
    hist = np.bincount(num_classes * t + p, minlength=num_classes**2)
    return hist.reshape(num_classes, num_classes)

def scores_from_cm(cm: np.ndarray):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    iou = tp / (tp + fp + fn + 1e-12)
    acc = tp / (tp + fn + 1e-12)
    oa  = tp.sum() / (cm.sum() + 1e-12)
    return {"OA": float(oa), "mAcc": float(np.nanmean(acc)), "mIoU": float(np.nanmean(iou))}

# ----------------- feature helpers -----------------
def estimate_normals_o3d(xyz: np.ndarray, k: int = 16) -> np.ndarray:
    """Estimate normals with Open3D if available; else zeros (keeps channel count)."""
    N = xyz.shape[0]
    if N == 0 or o3d is None:
        return np.zeros((N,3), dtype=np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    try:
        pcd.orient_normals_consistent_tangent_plane(k)
    except Exception:
        pass
    return np.asarray(pcd.normals, dtype=np.float32)

# ----------------- build full features (RGB and/or normals) -----------------

def build_features_full(xyz_np: np.ndarray, rgb_np: Optional[np.ndarray],
                        use_rgb: bool, use_normals: bool) -> Optional[np.ndarray]:
    """Return (N,C_extra) with RGB and/or normals, or None if no extras."""
    feats = None
    if use_rgb and (rgb_np is not None):
        feats = rgb_np.astype(np.float32)
    if use_normals:
        nrms = estimate_normals_o3d(xyz_np).astype(np.float32)
        feats = nrms if feats is None else np.concatenate([feats, nrms], axis=1)
    return feats

# ----------------- occlusion -----------------
# Sector drop in XY plane (robust, never empty)
# Points are assumed centered in XY (mean subtracted)
# This function randomly removes a wedge-shaped sector of points in the XY-plane to simulate occlusion, guaranteeing that at least a small portion of points always remain.


def random_sector_mask(points_centered_xy: np.ndarray, occlusion_pct: float, seed=0) -> np.ndarray:
    """Keep-mask after dropping a polar sector in XY (robust and never empty)."""
    N = len(points_centered_xy)
    if occlusion_pct <= 0 or N == 0:
        return np.ones(N, dtype=bool)
    rng = np.random.default_rng(seed)

    # Convert Cartesian XY to polar angles theta for each point (range -π to π).
    theta = np.arctan2(points_centered_xy[:,1], points_centered_xy[:,0])  # [-pi, pi]
    
    # Sector width corresponds to requested occlusion percentage; start angle chosen randomly.
    # Based on these, determine which points fall within the sector to drop.
    # Actually it gets an area of slice 2*width/pi (since width is half-angle), out of 360 degrees.
    width = 2*np.pi * (occlusion_pct / 100.0)
    start = rng.uniform(-np.pi, np.pi)
    end   = start + width
    if end <= np.pi:
        drop = (theta >= start) & (theta <= end)
    else:
        drop = (theta >= start) | (theta <= (end - 2*np.pi))
    keep = ~drop

    if not np.any(keep):  # guarantee non-empty
        nmin = max(1, int(0.01 * N))
        idx = rng.choice(N, size=nmin, replace=False)
        keep = np.zeros(N, dtype=bool); keep[idx] = True
    return keep

# ----------------- PointNet++ (SSG) — light CPU version: Similar implementation as eval_list or infer_pointnet2_cpu -----------------
def fps(x: torch.Tensor, m: int) -> torch.Tensor:
    B, N, _ = x.shape
    m = min(m, N)
    device = x.device
    idx = torch.zeros(B, m, dtype=torch.long, device=device)
    if N == 0 or m == 0:
        return idx
    farthest = torch.randint(0, N, (B,), device=device)
    dist = torch.full((B, N), 1e10, device=device)
    batch_indices = torch.arange(B, device=device)
    for i in range(m):
        idx[:, i] = farthest
        centroid = x[batch_indices, farthest, :].unsqueeze(1)
        d = torch.sum((x - centroid) ** 2, dim=2)
        dist = torch.minimum(dist, d)
        farthest = torch.max(dist, dim=1).indices
    return idx

def ball_query(x: torch.Tensor, centroids: torch.Tensor, radius: float, K: int) -> torch.Tensor:
    B, N = x.shape[0], x.shape[1]
    M = centroids.shape[1]
    K = min(K, N)
    if K == 0 or N == 0 or M == 0:
        return torch.zeros((B, M, 0), dtype=torch.long, device=x.device)
    dists = torch.cdist(centroids, x, p=2)             # (B,M,N)
    within = dists <= radius
    dists_masked = dists.clone()
    dists_masked[~within] = 1e10
    idx = torch.topk(dists_masked, k=K, dim=2, largest=False).indices  # (B,M,K)
    return idx

def index_points(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    B, N, C = x.shape
    batch_indices = torch.arange(B, device=x.device).view(B, 1, 1).expand(-1, idx.size(1), idx.size(2))
    return x[batch_indices, idx, :]

def three_nn_interpolate(xyz_src: torch.Tensor, xyz_dst: torch.Tensor, feat_dst: torch.Tensor) -> torch.Tensor:
    d = torch.cdist(xyz_src, xyz_dst, p=2) + 1e-8
    idx = torch.topk(d, k=3, dim=2, largest=False).indices
    d3 = torch.gather(d, 2, idx)
    w = (1.0 / d3)
    w = w / torch.sum(w, dim=2, keepdim=True)  # (B,N,3)
    w = w.unsqueeze(1)                         # (B,1,N,3)
    feat_dst_perm = feat_dst.permute(0, 2, 1)  # (B,M,C)
    gathered = index_points(feat_dst_perm, idx)   # (B,N,3,C)
    gathered = gathered.permute(0, 3, 1, 2)       # (B,C,N,3)
    return torch.sum(gathered * w, dim=3)         # (B,C,N)

class SA_SSG(nn.Module):
    def __init__(self, npoint: int, radius: float, nsample: int, in_channels: int, mlp: List[int]):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        layers = []
        last_c = in_channels + 3   # extra features + relative xyz
        for out_c in mlp:
            layers += [nn.Conv2d(last_c, out_c, 1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)]
            last_c = out_c
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz: torch.Tensor, features: Optional[torch.Tensor]):
        B, N, _ = xyz.shape
        npoint = min(self.npoint, N)
        if npoint == 0:
            return xyz[:, :0, :], torch.zeros(B, 0, 0, device=xyz.device)
        idx = fps(xyz, npoint)                                  # (B, npoint)
        new_xyz = xyz[torch.arange(B).unsqueeze(-1), idx]        # (B, npoint, 3)
        knn_idx = ball_query(xyz, new_xyz, self.radius, self.nsample)  # (B,npoint,K<=N)

        grouped_xyz = index_points(xyz, knn_idx)                 # (B,npoint,K,3)
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

        if features is not None:
            feat = features.permute(0, 2, 1)                     # (B,N,C)
            grouped_feat = index_points(feat, knn_idx)           # (B,npoint,K,C)
            new_feat = torch.cat([grouped_xyz_norm, grouped_feat], dim=-1)
        else:
            new_feat = grouped_xyz_norm

        new_feat = new_feat.permute(0, 3, 1, 2).contiguous()     # (B,C_in+3,npoint,K)
        new_feat = self.mlp(new_feat)                            # (B,C_out,npoint,K)
        new_feat = torch.max(new_feat, dim=3).values             # (B,C_out,npoint)
        return new_xyz, new_feat

class FP(nn.Module):
    def __init__(self, in_channels: int, mlp: List[int]):
        super().__init__()
        layers = []
        last_c = in_channels
        for out_c in mlp:
            layers += [nn.Conv1d(last_c, out_c, 1), nn.BatchNorm1d(out_c), nn.ReLU(inplace=True)]
            last_c = out_c
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz_src, xyz_dst, feat_src, feat_dst):
        interpolated = three_nn_interpolate(xyz_src, xyz_dst, feat_dst)  # (B,Cd,N)
        fused = torch.cat([interpolated, feat_src], dim=1) if feat_src is not None else interpolated
        return self.mlp(fused)

class PointNet2_SSG_Seg(nn.Module):
    def __init__(self, in_channels: int = 0, num_classes: int = 13) -> None:
        super().__init__()
        # Lighter CPU config (match widths commonly used)
        self.sa1 = SA_SSG(512,  0.10, 16, in_channels, [64, 64, 128])
        self.sa2 = SA_SSG(128,  0.20, 16, 128,         [128, 128, 256])
        self.sa3 = SA_SSG(32,   0.40, 16, 256,         [256, 256, 512])

        self.fp3 = FP(512 + 256, [256, 256])
        self.fp2 = FP(256 + 128, [256, 128])
        self.fp1 = FP(128 + in_channels, [128, 128, 128])

        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, xyz, feats: Optional[torch.Tensor] = None):
        l1_xyz, l1_feat = self.sa1(xyz, feats)
        l2_xyz, l2_feat = self.sa2(l1_xyz, l1_feat)
        l3_xyz, l3_feat = self.sa3(l2_xyz, l2_feat)

        l2_up = self.fp3(l2_xyz, l3_xyz, l2_feat, l3_feat)
        l1_up = self.fp2(l1_xyz, l2_xyz, l1_feat, l2_up)

        up0 = self.fp1(xyz, l1_xyz, feats, l1_up)
        return self.head(up0)   # (B,num_classes,N)

# ----------------- chunked inference -----------------
@torch.no_grad()
def infer_chunks(model: nn.Module, xyz_np: np.ndarray,
                 feats_np_or_none: Optional[np.ndarray], device: torch.device,
                 num_classes: int, chunk_size: int) -> np.ndarray:
    """
    xyz_np: (N,3)
    feats_np_or_none: (N,C_extra) or None
    Returns pred: (N,) int64
    """
    N = xyz_np.shape[0]
    pred = np.empty(N, dtype=np.int64)
    i = 0
    while i < N:
        j = min(i + chunk_size, N)
        xyz = torch.from_numpy(xyz_np[i:j]).float().unsqueeze(0).to(device)      # (1,M,3)
        feats = None
        if feats_np_or_none is not None:
            feats = torch.from_numpy(feats_np_or_none[i:j]).float().unsqueeze(0).permute(0,2,1).to(device)  # (1,C,M)
        logits = model(xyz, feats)                                               # (1,Cnum,M)
        if logits.shape[1] == num_classes:
            logits = logits.permute(0,2,1).contiguous()                          # (1,M,Cnum)
        pred[i:j] = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
        i = j
    return pred

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser("Occlusion robustness eval (multi-seed)")
    ap.add_argument("--ply", required=True, help="input PLY with xyz [rgb] [label]")
    ap.add_argument("--ckpt", required=True, help="checkpoint (.pth) for PointNet++")
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--use_rgb", action="store_true", help="must match training (RGB as extra features)")
    ap.add_argument("--use_normals", action="store_true", help="estimate/use normals as extra features (match training)")
    ap.add_argument("--occlusion", default="0,20,40,60,80", help="comma list of percentages")
    ap.add_argument("--chunk_size", type=int, default=80000, help="points per chunk to avoid OOM")
    ap.add_argument("--seeds", type=str, default="42", help="comma-separated seeds, e.g. 1,2,3,4,5")
    ap.add_argument("--out_csv", type=str, default="", help="optional: path to save mean±std table")
    args = ap.parse_args()

    # Load scene (PLY or sidecar labels)
    rec = _parse_ascii_ply(args.ply)
    xyz_np = rec["xyz"].astype(np.float32)            # (N,3)
    rgb_np = rec["rgb"].astype(np.float32) if (rec.get("rgb") is not None) else None

    gt_np = rec.get("label", None)
    if gt_np is None:
        gt_np = _load_sidecar_labels(args.ply)
    if gt_np is None:
        raise KeyError("Ground-truth labels not found. Add 'label' to PLY or provide sidecar file "
                       "('*_gt.txt', '*_labels.txt', '*.labels', or CSV with a 'label' column').")
    gt_np = gt_np.astype(np.int64)
    if gt_np.shape[0] != xyz_np.shape[0]:
        raise ValueError(f"Label length ({gt_np.shape[0]}) != points ({xyz_np.shape[0]}).")

    # Center in XY for stable sectoring
    xy_mean = xyz_np[:, :2].mean(axis=0, keepdims=True)
    xyz_centered = xyz_np.copy()
    xyz_centered[:, :2] -= xy_mean

    # Pre-build full-scene extra features once (RGB and/or normals)
    full_feats = build_features_full(xyz_np, rgb_np, args.use_rgb, args.use_normals)  # (N,C_extra) or None

    # Build model
    device = torch.device("cpu")
    feat_ch = (3 if (args.use_rgb and (rgb_np is not None)) else 0) + (3 if args.use_normals else 0)
    model = PointNet2_SSG_Seg(in_channels=feat_ch, num_classes=args.num_classes).to(device)

    # Robust checkpoint load with clear validation of channel config
    sd_raw = torch.load(args.ckpt, map_location=device)
    # unwrap state_dict if nested
    if isinstance(sd_raw, dict):
        sd = None
        for key in ("state_dict", "model_state_dict", "model"):
            if key in sd_raw and isinstance(sd_raw[key], dict):
                sd = sd_raw[key]; break
        if sd is None:
            sd = sd_raw
    else:
        sd = sd_raw

    # Clean common prefixes
    cleaned = {k.replace("module.", "").replace("model.", ""): v for k, v in sd.items()}

    # Validate: first SA conv expects (3 + feat_ch) input channels
    # Find sa1 conv weight
    sa1_key = None
    for cand in ("sa1.mlp.0.weight",):
        if cand in cleaned:
            sa1_key = cand; break
    if sa1_key is None:
        # Allow loading; if it fails, PyTorch will show which key is missing
        pass
    else:
        expected_total = cleaned[sa1_key].shape[1]    # channels seen during training (XYZ + extras)
        current_total  = 3 + feat_ch
        if expected_total != current_total:
            raise RuntimeError(
                f"Checkpoint expects total input channels {expected_total} (XYZ+extras), "
                f"but current config provides {current_total}. "
                f"Hint: toggle --use_rgb/--use_normals to match training. "
                f"(Training likely used extras={expected_total - 3} → set flags accordingly.)"
            )

    model.load_state_dict(cleaned, strict=True) # require exact match
    model.eval()            # Test mode, no dropout, batchnorm fix

    occl_list = [float(s.strip()) for s in args.occlusion.split(",") if s.strip()!=""]      # occlusion percentages
    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()!=""]            # random seeds

    # collect per-seed results
    results = {pct: [] for pct in occl_list}    # pct -> list of (num_points, OA, mAcc, mIoU)

    # run across seeds
    for seed in seed_list:
        np.random.seed(seed); torch.manual_seed(seed)
        for pct in occl_list:
            keep = random_sector_mask(xyz_centered, pct, seed=seed)
            xyz_k = xyz_np[keep]
            feats_k = None if full_feats is None else full_feats[keep]
            gt_k  = gt_np[keep]
            pred_k = infer_chunks(model, xyz_k, feats_k, device, args.num_classes, args.chunk_size)
            cm = confusion_matrix(args.num_classes, gt_k, pred_k)
            sc = scores_from_cm(cm)
            results[pct].append( (len(xyz_k), sc["OA"], sc["mAcc"], sc["mIoU"]) )

    # print mean±std table
    print("occlusion_pct,num_points_mean,num_points_std,OA_mean,OA_std,mAcc_mean,mAcc_std,mIoU_mean,mIoU_std")
    for pct in occl_list:
        arr = np.array(results[pct], dtype=float)  # shape (S, 4)
        n_mean, n_std   = arr[:,0].mean(), arr[:,0].std(ddof=0)
        oa_mean, oa_std = arr[:,1].mean(), arr[:,1].std(ddof=0)
        ma_mean, ma_std = arr[:,2].mean(), arr[:,2].std(ddof=0)
        mi_mean, mi_std = arr[:,3].mean(), arr[:,3].std(ddof=0)
        print(f"{pct:.0f},{int(round(n_mean))},{int(round(n_std))},"
              f"{oa_mean:.4f},{oa_std:.4f},{ma_mean:.4f},{ma_std:.4f},{mi_mean:.4f},{mi_std:.4f}")

    # optional: save CSV
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["occlusion_pct","num_points_mean","num_points_std",
                         "OA_mean","OA_std","mAcc_mean","mAcc_std","mIoU_mean","mIoU_std"])
            for pct in occl_list:
                arr = np.array(results[pct], dtype=float)
                n_mean, n_std   = arr[:,0].mean(), arr[:,0].std(ddof=0)
                oa_mean, oa_std = arr[:,1].mean(), arr[:,1].std(ddof=0)
                ma_mean, ma_std = arr[:,2].mean(), arr[:,2].std(ddof=0)
                mi_mean, mi_std = arr[:,3].mean(), arr[:,3].std(ddof=0)
                wr.writerow([pct, int(round(n_mean)), int(round(n_std)),
                             f"{oa_mean:.4f}", f"{oa_std:.4f}",
                             f"{ma_mean:.4f}", f"{ma_std:.4f}",
                             f"{mi_mean:.4f}", f"{mi_std:.4f}"])

if __name__ == "__main__":
    main()
