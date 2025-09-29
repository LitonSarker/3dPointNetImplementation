#!/usr/bin/env python3
# eval_list_single.py  —  One-file, chunk-safe evaluator for PointNet++ (CPU)
# Usage:
#   python eval_list_single.py --files .\out_ply\val_full.txt --num_classes 13 \
#     --ckpt runs\seg_cpu\checkpoints\best_model.pth --use_rgb --use_normals \
#     --chunk_size 120000 --out_csv .\results\eval_val_full.csv

import argparse, os, csv
import numpy as np
import torch # type: ignore
import torch.nn as nn

# Optional (used by the PLY fallback path and normals)
try:
    import open3d as o3d
except Exception:
    o3d = None

def estimate_normals_o3d(xyz: np.ndarray, k: int = 16) -> np.ndarray:
    """Estimate normals; returns (N,3) float32. Falls back to zeros if Open3D unavailable."""
    N = xyz.shape[0]
    if N == 0:
        return np.zeros((0,3), dtype=np.float32)
    if o3d is None:
        # Fallback: zeros (keeps channel count correct for checkpoints that expect normals)
        return np.zeros((N,3), dtype=np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    try:
        pcd.orient_normals_consistent_tangent_plane(k)
    except Exception:
        pass
    return np.asarray(pcd.normals, dtype=np.float32)

# ===========================
# 1) Minimal ASCII PLY parser
# ===========================
def _parse_ascii_ply(path: str):
    """
    Parses ASCII .ply with header containing 'element vertex N' and per-vertex properties.
    Supports: x y z [red green blue] [label]
    Returns dict: {'xyz': (N,3) float32, 'rgb': (N,3) float32 or None, 'label': (N,) int64 or None}
    """
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        header = []
        line = f.readline()
        if not line.startswith('ply'):
            raise ValueError(f'Not a PLY file: {path}')
        header.append(line)
        num_vertices = None
        properties = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError('Unexpected EOF in header')
            header.append(line)
            s = line.strip()
            if s.startswith('element vertex'):
                num_vertices = int(s.split()[-1])
            if s.startswith('property'):
                parts = s.split()
                if len(parts) >= 3:
                    properties.append(parts[-1].lower())
            if s == 'end_header':
                break

        data = []
        for _ in range(int(num_vertices or 0)):
            ln = f.readline()
            if not ln:
                break
            parts = ln.strip().split()
            if parts:
                data.append(parts)

    if len(data) == 0:
        # Fallback via Open3D (if available)
        if o3d is None:
            raise ValueError(f"No vertex data parsed and Open3D not available for {path}")
        pcd = o3d.io.read_point_cloud(path)
        xyz = np.asarray(pcd.points, dtype=np.float32)
        rgb = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None
        return {'xyz': xyz, 'rgb': rgb, 'label': None}

    arr = np.asarray(data, dtype=np.float32)
    name_to_col = {name: idx for idx, name in enumerate(properties[:arr.shape[1]])}
    # xyz
    x_idx = name_to_col.get('x', 0)
    y_idx = name_to_col.get('y', 1)
    z_idx = name_to_col.get('z', 2)
    xyz = arr[:, [x_idx, y_idx, z_idx]].astype(np.float32)

    # rgb
    r_key = 'red' if 'red' in name_to_col else ('r' if 'r' in name_to_col else None)
    g_key = 'green' if 'green' in name_to_col else ('g' if 'g' in name_to_col else None)
    b_key = 'blue' if 'blue' in name_to_col else ('b' if 'b' in name_to_col else None)
    if r_key and g_key and b_key:
        rgb = arr[:, [name_to_col[r_key], name_to_col[g_key], name_to_col[b_key]]].astype(np.float32)
        if rgb.size and rgb.max() > 1.0:
            rgb = rgb / 255.0
    else:
        rgb = None
        if o3d is not None:
            try:
                pcd = o3d.io.read_point_cloud(path)
                if pcd.has_colors():
                    rgb = np.asarray(pcd.colors, dtype=np.float32)
            except Exception:
                pass

    # label
    lbl = None
    for cand in ('label', 'seg_label', 'class', 'y'):
        if cand in name_to_col:
            lbl = arr[:, name_to_col[cand]].astype(np.int64)
            break

    return {'xyz': xyz, 'rgb': rgb, 'label': lbl}

# ===========================
# 2) CPU metrics (cm, scores)
# ===========================
def confusion_matrix(num_classes: int, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    mask = (gt >= 0) & (gt < num_classes)
    gt = gt[mask]
    pred = pred[mask]
    for g, p in zip(gt, pred):
        cm[g, p] += 1
    return cm

def scores_from_cm(cm: np.ndarray):
    eps = 1e-9
    total = cm.sum()
    correct = np.trace(cm)
    oa = correct / (total + eps)
    tp = np.diag(cm).astype(np.float64)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp

    denom = tp + fp + fn + eps
    iou = tp / denom
    acc_per = tp / (tp + fn + eps)

    return {
        'OA': float(oa),
        'mAcc': float(np.mean(acc_per)),
        'mIoU': float(np.mean(iou)),
        'IoU_per_class': [float(x) for x in iou],
        'Acc_per_class': [float(x) for x in acc_per],
    }

# =========================================
# 3) PointNet++ (SSG) — CPU-friendly module
# =========================================

def fps(x: torch.Tensor, m: int) -> torch.Tensor:
    B, N, _ = x.shape
    m = min(m, N)                           # clamp
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
    # x: (B,N,3), centroids: (B,M,3)
    B, N = x.shape[0], x.shape[1]
    M = centroids.shape[1]
    K = min(K, N)                           # clamp
    if K == 0 or N == 0 or M == 0:
        return torch.zeros((B, M, 0), dtype=torch.long, device=x.device)
    dists = torch.cdist(centroids, x, p=2)  # (B,M,N)
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
    w = w / torch.sum(w, dim=2, keepdim=True)     # (B,N,3)
    w = w.unsqueeze(1)                            # (B,1,N,3)
    feat_dst_perm = feat_dst.permute(0, 2, 1)     # (B,M,C)
    gathered = index_points(feat_dst_perm, idx)   # (B,N,3,C)
    gathered = gathered.permute(0, 3, 1, 2)       # (B,C,N,3)
    return torch.sum(gathered * w, dim=3)         # (B,C,N)

class SA_SSG(nn.Module):
    def __init__(self, npoint: int, radius: float, nsample: int, in_channels: int, mlp):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        layers = []
        last_c = in_channels + 3
        for out_c in mlp:
            layers += [nn.Conv2d(last_c, out_c, 1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)]
            last_c = out_c
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor | None):
        B, N, _ = xyz.shape
        npoint = min(self.npoint, N)            # clamp
        if npoint == 0:
            # Degenerate chunk guard
            return xyz[:, :0, :], torch.zeros(B, 0, 0, device=xyz.device)

        idx = fps(xyz, npoint)                  # (B, npoint)
        new_xyz = xyz[torch.arange(B).unsqueeze(-1), idx]  # (B, npoint, 3)
        knn_idx = ball_query(xyz, new_xyz, self.radius, self.nsample)  # (B,npoint,K<=N)

        grouped_xyz = index_points(xyz, knn_idx)                       # (B,npoint,K,3)
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

        if features is not None:
            feat = features.permute(0, 2, 1)                           # (B,N,C)
            grouped_feat = index_points(feat, knn_idx)                 # (B,npoint,K,C)
            new_feat = torch.cat([grouped_xyz_norm, grouped_feat], dim=-1)
        else:
            new_feat = grouped_xyz_norm

        new_feat = new_feat.permute(0, 3, 1, 2).contiguous()           # (B,C_in+3,npoint,K)
        new_feat = self.mlp(new_feat)                                  # (B,C_out,npoint,K)
        new_feat = torch.max(new_feat, dim=3).values                   # (B,C_out,npoint)
        return new_xyz, new_feat

class FP(nn.Module):
    def __init__(self, in_channels: int, mlp):
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
    def __init__(self, in_channels: int = 3, num_classes: int = 4) -> None:
        super().__init__()
        # Lighter settings for CPU
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

    def forward(self, xyz, feats: torch.Tensor | None = None):
        l1_xyz, l1_feat = self.sa1(xyz, feats)
        l2_xyz, l2_feat = self.sa2(l1_xyz, l1_feat)
        l3_xyz, l3_feat = self.sa3(l2_xyz, l2_feat)

        l2_up = self.fp3(l2_xyz, l3_xyz, l2_feat, l3_feat)
        l1_up = self.fp2(l1_xyz, l2_xyz, l1_feat, l2_up)

        skip0 = feats if feats is not None else None
        up0 = self.fp1(xyz, l1_xyz, skip0, l1_up)

        return self.head(up0)   # (B,num_classes,N)

# ==================================
# 4) Chunked inference & CLI runner
# ==================================
@torch.no_grad()
def infer_one_chunked(model, rec, num_classes, use_rgb, use_normals, device, chunk_size=60000):
    xyz_np = rec["xyz"]                                # (N,3)
    labels = rec["label"].astype(np.int64) if rec.get("label") is not None else None

    # --- build feature channels (RGB, normals) ---
    feats_np = None
    if use_rgb and ("rgb" in rec) and (rec["rgb"] is not None):
        feats_np = rec["rgb"].astype(np.float32)
    if use_normals:
        try:
            nrms = estimate_normals_o3d(xyz_np).astype(np.float32)
        except Exception:
            nrms = np.zeros((xyz_np.shape[0], 3), dtype=np.float32)
        feats_np = nrms if feats_np is None else np.concatenate([feats_np, nrms], axis=1)
        # ... after you finish computing feats_np from RGB / normals ...
    # Auto-pad features to match checkpoint's expected in_channels (from SA1)
    try:
        exp_in = int(model.sa1.mlp[0].weight.shape[1] - 3)  # expected feature channels (excl. xyz)
    except Exception:
        exp_in = 0

    cur_in = 0 if feats_np is None else feats_np.shape[1]
    if exp_in > 0 and cur_in < exp_in:
        pad = np.zeros((xyz_np.shape[0], exp_in - cur_in), dtype=np.float32)
        feats_np = pad if feats_np is None else np.concatenate([feats_np, pad], axis=1)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    N  = xyz_np.shape[0]
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        xyz = torch.from_numpy(xyz_np[s:e]).float().unsqueeze(0).to(device)  # (1,n,3)
        if feats_np is None:
            feats = None
        else:
            feats = torch.from_numpy(feats_np[s:e]).float().unsqueeze(0).permute(0,2,1).to(device)  # (1,C,n)

        logits = model(xyz, feats)                                  # (1,C,n)
        if logits.shape[1] == num_classes:
            logits = logits.permute(0,2,1).contiguous()             # (1,n,C)
        pred = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
        if labels is not None:
            cm += confusion_matrix(num_classes, labels[s:e], pred)
    return cm

# --- auto-detect expected input channels from checkpoint ---
def _expected_in_ch_from_sd(state):
    # sa1.mlp.0.weight has shape [64, (in_ch+3), 1, 1]
    w = state.get("sa1.mlp.0.weight", None)
    if w is None:
        # try to find by suffix if keys are prefixed
        for k, v in state.items():
            if k.endswith("sa1.mlp.0.weight"):
                w = v; break
    if w is not None and w.ndim == 4:
        return int(w.shape[1] - 3)
    return None


def main():
    ap = argparse.ArgumentParser(description="Evaluate PointNet++ on a list of PLY files (chunk-safe, single file)")
    ap.add_argument('--files', type=str, required=True, help='txt file with one PLY path per line')
    ap.add_argument('--num_classes', type=int, required=True)
    ap.add_argument('--ckpt', type=str, required=True, help='checkpoint path (state_dict)')
    ap.add_argument('--use_rgb', action='store_true', help='set only if you trained with RGB and PLY has rgb fields')
    ap.add_argument('--use_normals', action='store_true', help='estimate/use normals as extra features')
    ap.add_argument('--chunk_size', type=int, default=60000, help='points per chunk to limit RAM')
    ap.add_argument('--out_csv', type=str, help='optional CSV path for overall & per-class metrics')
    ap.add_argument('--class_names', type=str, help='optional comma-separated names for classes (length=num_classes)')
    args = ap.parse_args()

    device = torch.device('cpu')

    # Read list
    with open(args.files, 'r', encoding='utf-8') as f:
        ply_files = [ln.strip() for ln in f if ln.strip()]
    if len(ply_files) == 0:
        raise RuntimeError("No PLY files found in --files list.")

    # Decide in_channels from the first file + flags (extra features only)
    first = _parse_ascii_ply(ply_files[0])
    C = 0
    if args.use_rgb and ('rgb' in first) and (first['rgb'] is not None):
        C += 3
    if args.use_normals:
        C += 3
    in_ch = C

    # Build & load model
    model = PointNet2_SSG_Seg(in_channels=in_ch, num_classes=args.num_classes).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    

    exp_in_ch = _expected_in_ch_from_sd(sd)
    if exp_in_ch is not None and exp_in_ch != in_ch:
        print(f"[INFO] Rebuilding model: checkpoint expects in_channels={exp_in_ch}, "
            f"but CLI requested in_channels={in_ch}. Using {exp_in_ch}.")
        in_ch = exp_in_ch
        model = PointNet2_SSG_Seg(in_channels=in_ch, num_classes=args.num_classes).to(device)


    model.load_state_dict(sd, strict=True)
    model.eval()

    # Class names (optional)
    if args.class_names:
        class_names = [s.strip() for s in args.class_names.split(",")]
        if len(class_names) != args.num_classes:
            raise ValueError("--class_names length must match --num_classes")
    else:
        class_names = [f"class_{i}" for i in range(args.num_classes)]

    # Aggregate CM across scenes
    cm_total = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    for p in ply_files:
        print(f"[Eval] {os.path.basename(p)}")
        rec = _parse_ascii_ply(p)
        cm_total += infer_one_chunked(
            model, rec, args.num_classes, args.use_rgb, args.use_normals, device, args.chunk_size
        )

    scores = scores_from_cm(cm_total)

    # Pretty print
    print("\n=== Final Evaluation (chunked) ===")
    for k, v in scores.items():
        if isinstance(v, (list, np.ndarray)):
            print(f"{k}: {[round(float(x),4) for x in v]}")
        else:
            print(f"{k}: {float(v):.4f}")

    # Optional CSV export
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Section","Metric","Value"])
            for k, v in scores.items():
                if isinstance(v, (list, np.ndarray)):
                    continue
                w.writerow(["overall", k, round(float(v), 6)])
            acc_per = scores.get("Acc_per_class", [])
            iou_per = scores.get("IoU_per_class", [])
            for i in range(args.num_classes):
                acc = float(acc_per[i]) if i < len(acc_per) else float("nan")
                iou = float(iou_per[i]) if i < len(iou_per) else float("nan")
                w.writerow(["per_class", class_names[i], f"acc={round(acc,6)};iou={round(iou,6)}"])
        print(f"[OK] wrote CSV → {args.out_csv}")

if __name__ == "__main__":
    main()
