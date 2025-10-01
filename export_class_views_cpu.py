#!/usr/bin/env python3
# export_class_views_cpu.py, CPU-only exporter of per-class point clouds + metrics
# Usage (single scene):
# Mentioned in the command line
# Notes:
# - Features are built as [XYZ, RGB, normals], then matched to the checkpoint's expected in_channels

import argparse, os, csv, json, collections
import numpy as np
import torch
import torch.nn as nn

# Optional (for PLY IO, normals, and smoothing)
try:
    import open3d as o3d
except Exception:
    o3d = None

# -----------------------
# Minimal ASCII PLY parser, it loads the .ply file 
# -----------------------
def parse_ascii_ply(path: str):
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
                    properties.append(parts[-1])
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
    def _rgb_from_arr():
        r_key = 'red' if 'red' in name_to_col else ('r' if 'r' in name_to_col else None)
        g_key = 'green' if 'green' in name_to_col else ('g' if 'g' in name_to_col else None)
        b_key = 'blue' if 'blue' in name_to_col else ('b' if 'b' in name_to_col else None)
        if r_key and g_key and b_key:
            rgb0 = arr[:, [name_to_col[r_key], name_to_col[g_key], name_to_col[b_key]]].astype(np.float32)
            if rgb0.size and rgb0.max() > 1.0:
                rgb0 = rgb0 / 255.0
            return rgb0
        return None

    rgb = _rgb_from_arr()
    if rgb is None and o3d is not None:
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

# -----------
# Metrics, getting the matrix genetation parameters to depict
# -----------
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

# ----------------------
# Eval-time conveniences
# ----------------------
# Re-centers the point cloud to the origin and scales it into a unit sphere. It prevents bias from absolute location/scale.
def normalize_xyz(xyz: np.ndarray) -> np.ndarray:
    c = xyz.mean(axis=0, keepdims=True)
    xyz0 = xyz - c
    r = np.linalg.norm(xyz0, axis=1).max() + 1e-9
    return xyz0 / r

# Uses Open3D to estimate surface normals for each point by fitting a tangent plane to its k nearest neighbors.
def estimate_normals_o3d(xyz: np.ndarray, k: int = 16) -> np.ndarray:
    if o3d is None or xyz.shape[0] == 0:
        return np.zeros((xyz.shape[0], 3), dtype=np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    try:
        pcd.orient_normals_consistent_tangent_plane(k)
    except Exception:
        pass
    return np.asarray(pcd.normals, dtype=np.float32)

# Post-processes noisy predicted labels by applying k-nearest neighbor (kNN) majority voting
# Smooths out isolated misclassifications and creates more spatially consistent segmentation maps.

def smooth_labels_knn(points_xyz: np.ndarray, labels: np.ndarray, k: int = 12) -> np.ndarray:
    if k <= 1 or o3d is None:
        return labels
    valid = labels >= 0
    if not np.any(valid):
        return labels
    pts = points_xyz[valid].astype(np.float64)
    lbl = labels[valid]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    kdt = o3d.geometry.KDTreeFlann(pcd)
    out = labels.copy()
    idx_map = np.where(valid)[0]
    for j in range(pts.shape[0]):
        ok, nn_idx, _ = kdt.search_knn_vector_3d(pts[j], k)
        if ok > 0:
            votes = lbl[np.asarray(nn_idx, int)]
            c = collections.Counter(votes.tolist()).most_common(1)[0][0]
            out[idx_map[j]] = c
    return out

# ----------------------
# PointNet++ (SSG) CPU: Similar implementation as eval_list or infer_pointnet2_cpu
# ----------------------
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
        npoint = min(self.npoint, N)
        if npoint == 0:
            return xyz[:, :0, :], torch.zeros(B, 0, 0, device=xyz.device)

        idx = fps(xyz, npoint)
        new_xyz = xyz[torch.arange(B).unsqueeze(-1), idx]  # (B, npoint, 3)
        knn_idx = ball_query(xyz, new_xyz, self.radius, self.nsample)  # (B,npoint,K)

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
    def __init__(self, in_channels: int = 3, num_classes: int = 13) -> None:
        super().__init__()
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

# ---------------
# PLY writer (xyzrgb): Writes a point cloud with 3D coordinates and RGB colors into an ASCII PLY file.
# ---------------
def write_ply_xyzrgb(path, xyz, rgb):
    n = xyz.shape[0]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = xyz[i]
            r, g, b = rgb[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

# ------------------------
# Feature composition (match checkpoint)
# ------------------------
# Builds the input feature matrix for PointNet++ by combining XYZ coordinates, RGB values, and optionally normals, then adjusts the number of channels to match what the checkpointed model expects
# This ensures that whatever features you build (XYZ, RGB, normals) are always compatible with the model’s expected input channels, avoiding dimension mismatch errors.

def compose_features(expected_in: int,
                     xyz_eval: np.ndarray,
                     rec: dict,
                     use_rgb: bool,
                     want_normals: bool,
                     knn_normals: int) -> np.ndarray | None:
    """
    Build features in canonical order [XYZ, RGB, normals] and then
    truncate/zero-pad to match 'expected_in' channels.
    """
    comps = []
    # Always try to start with XYZ
    comps.append(xyz_eval.astype(np.float32))                     # (N,3)

    # Optional RGB
    if use_rgb and ('rgb' in rec) and (rec['rgb'] is not None):
        comps.append(rec['rgb'].astype(np.float32))               # (N,3) in 0..1

    # Optional normals (estimated)
    if want_normals:
        nrms = estimate_normals_o3d(xyz_eval, k=knn_normals).astype(np.float32)
        comps.append(nrms)                                        # (N,3)

    # Concatenate what we have
    feats_full = np.concatenate(comps, axis=1) if len(comps) else None
    if feats_full is None:
        return None

    C = feats_full.shape[1]
    if C == expected_in:
        return feats_full

    # Truncate or pad with zeros to match expected_in
    if C > expected_in:
        feats_adj = feats_full[:, :expected_in]
        print(f"[INFO] Truncated features from {C} to {expected_in} channels (order=XYZ,RGB,normals).")
    else:
        pad = np.zeros((feats_full.shape[0], expected_in - C), dtype=np.float32)
        feats_adj = np.concatenate([feats_full, pad], axis=1)
        print(f"[INFO] Padded features from {C} to {expected_in} channels with zeros.")
    return feats_adj

# ------------------------
# Inference helpers
# ------------------------
@torch.no_grad()
def infer_scene_chunked(model, rec, args, device, expected_in, scene_name="scene"):
    xyz_raw = rec['xyz'].astype(np.float32)
    xyz_eval = normalize_xyz(xyz_raw) if args.normalize else xyz_raw

    # Build features to match checkpoint's expected_in
    feats_np = compose_features(
        expected_in=expected_in,
        xyz_eval=xyz_eval,
        rec=rec,
        use_rgb=args.use_rgb,
        want_normals=args.use_normals,   # only used if expected_in requires it (else truncated)
        knn_normals=args.knn_normals
    )

    N = xyz_eval.shape[0]
    pred = np.empty(N, dtype=np.int64)

    for s in range(0, N, args.chunk_size):
        e = min(s + args.chunk_size, N)
        if args.progress:
            print(f"[{scene_name}]  chunk {s}:{e}/{N}")
        xyz = torch.from_numpy(xyz_eval[s:e]).float().unsqueeze(0).to(device)  # (1,m,3)
        feats = None if feats_np is None else torch.from_numpy(feats_np[s:e]).float().unsqueeze(0).permute(0,2,1).to(device)

        logits = model(xyz, feats)  # (1,C,m)
        # Model's head returns (1,num_classes,N); take argmax over class dim=1
        pred_chunk = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()   # -> (m,)

        if pred_chunk.shape[0] != (e - s):
            raise RuntimeError(f"Chunk prediction length mismatch: got {pred_chunk.shape[0]} vs {(e - s)}")
        pred[s:e] = pred_chunk

    # Optional smoothing in original space
    if args.smooth_knn and args.smooth_knn > 0:
        pred = smooth_labels_knn(xyz_raw, pred, args.smooth_knn)

    return pred, xyz_raw

# ------------------------
# Main
# ------------------------
def main():
    ap = argparse.ArgumentParser(description="Export per-class PLYs + class-wise metrics (CPU, chunk-safe)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--files', type=str, help='txt with one PLY path per line')
    g.add_argument('--ply',   type=str, help='single PLY path')
    ap.add_argument('--ckpt', type=str, required=True, help='model checkpoint (.pth or dict with state_dict)')
    ap.add_argument('--num_classes', type=int, required=True)
    ap.add_argument('--outdir', type=str, required=True)

    # features / preprocessing (we’ll still match the checkpoint automatically)
    ap.add_argument('--use_rgb', action='store_true', help='include RGB channels if present')
    ap.add_argument('--use_normals', action='store_true', help='estimate normals via Open3D (may be truncated)')
    ap.add_argument('--knn_normals', type=int, default=12, help='K for normal estimation')
    ap.add_argument('--normalize', action='store_true', help='center & scale XYZ at eval (match train if used)')
    ap.add_argument('--smooth_knn', type=int, default=0, help='k-NN majority smoothing (0=off)')

    # runtime
    ap.add_argument('--chunk_size', type=int, default=60000)
    ap.add_argument('--progress', action='store_true')
    ap.add_argument('--class_names', type=str, help='comma-separated labels (len=num_classes)')
    args = ap.parse_args()

    # Resolve scenes
    if args.files:
        with open(args.files, 'r', encoding='utf-8-sig') as f:
            ply_files = [ln.strip().lstrip('\ufeff') for ln in f if ln.strip()]
    else:
        ply_files = [args.ply]
    if len(ply_files) == 0:
        raise RuntimeError("No PLY files to process.")

    # Class names
    if args.class_names:
        class_names = [s.strip() for s in args.class_names.split(",")]
        if len(class_names) != args.num_classes:
            raise ValueError("--class_names length must match --num_classes")
    else:
        class_names = [f"class_{i}" for i in range(args.num_classes)]

    device = torch.device('cpu')

    # ---- Load checkpoint & detect expected_in (feature channels) ----
    sd = torch.load(args.ckpt, map_location=device)
    # unwrap common wrappers
    cand = None
    if isinstance(sd, dict):
        for k in ("state_dict", "model_state_dict", "model"):
            if k in sd and isinstance(sd[k], dict):
                cand = sd[k]; break
    sd_use = cand if cand is not None else sd

    expected_in = None
    for k, v in sd_use.items():
        # first SA conv: weight shape [64, in_channels+3, 1, 1]
        if k.endswith("sa1.mlp.0.weight") and hasattr(v, "shape") and len(v.shape) == 4:
            expected_in = int(v.shape[1] - 3)
            break
    if expected_in is None:
        # fallback: assume common setting XYZ+RGB
        expected_in = 6
        print("[WARN] Could not infer in_channels from checkpoint; defaulting to 6 (XYZ+RGB).")

    # Build model with expected_in
    model = PointNet2_SSG_Seg(in_channels=expected_in, num_classes=args.num_classes).to(device)

    # Load weights (cleanup prefixes)
    cleaned = {}
    for k, v in sd_use.items():
        nk = k.replace("model.", "").replace("module.", "")
        cleaned[nk] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if len(missing) or len(unexpected):
        print(f"[CKPT] loaded with missing={len(missing)} unexpected={len(unexpected)}")
    print(f"[INFO] Checkpoint expects in_channels={expected_in}.")
    model.eval()

    # Prepare outputs
    os.makedirs(args.outdir, exist_ok=True)
    per_scene_csv = os.path.join(args.outdir, "per_scene_metrics.csv")
    global_csv    = os.path.join(args.outdir, "dataset_metrics.csv")
    legend_json   = os.path.join(args.outdir, "class_legend.json")

    # Palette
    BASE_PALETTE = np.array([
        [ 31,119,180],[255,127, 14],[ 44,160, 44],[214, 39, 40],[148,103,189],
        [140, 86, 75],[227,119,194],[127,127,127],[188,189, 34],[ 23,190,207],
        [174,199,232],[255,187,120],[152,223,138]
    ], dtype=np.uint8)
    def make_palette(n):
        if n <= BASE_PALETTE.shape[0]:
            return BASE_PALETTE[:n].copy()
        extra = n - BASE_PALETTE.shape[0]
        hues = (np.arange(extra) / max(1, extra)).reshape(-1, 1)
        s = np.full((extra,1), 0.65, dtype=np.float32)
        v = np.full((extra,1), 0.95, dtype=np.float32)
        h = hues[:,0]; S = s[:,0]; V = v[:,0]
        i = np.floor(h * 6).astype(int)
        f = h * 6 - i
        p = V * (1 - S)
        q = V * (1 - f * S)
        t = V * (1 - (1 - f) * S)
        i_mod = i % 6
        r = np.select([i_mod==0, i_mod==1, i_mod==2, i_mod==3, i_mod==4, i_mod==5],
                      [V, q, p, p, t, V], default=0)
        g = np.select([i_mod==0, i_mod==1, i_mod==2, i_mod==3, i_mod==4, i_mod==5],
                      [t, V, V, q, p, p], default=0)
        b = np.select([i_mod==0, i_mod==1, i_mod==2, i_mod==3, i_mod==4, i_mod==5],
                      [p, p, t, V, V, q], default=0)
        extra_rgb = np.clip(np.round(np.stack([r,g,b], axis=1)*255.0),0,255).astype(np.uint8)
        return np.vstack([BASE_PALETTE, extra_rgb])

    PALETTE = make_palette(args.num_classes)

    # Legend JSON
    with open(legend_json, "w", encoding="utf-8") as jf:
        json.dump({
            "num_classes": args.num_classes,
            "classes": [{"id": c, "name": (class_names[c] if c < len(class_names) else f'class_{c}'), "color_rgb": PALETTE[c].tolist()} for c in range(args.num_classes)]
        }, jf, indent=2)

    # CSV headers
    headers = ["scene", "points_total"] + \
              [f"count_{class_names[c] if c < len(class_names) else f'class_{c}'}" for c in range(args.num_classes)] + \
              ["OA", "mAcc", "mIoU"] + \
              [f"IoU_{class_names[c] if c < len(class_names) else f'class_{c}'}" for c in range(args.num_classes)] + \
              [f"Acc_{class_names[c] if c < len(class_names) else f'class_{c}'}" for c in range(args.num_classes)]

    with open(per_scene_csv, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(headers)

        cm_total = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)

        for ply_path in ply_files:
            base = os.path.splitext(os.path.basename(ply_path))[0]
            out_scene_dir = os.path.join(args.outdir, base)
            os.makedirs(out_scene_dir, exist_ok=True)
            print(f"[Export] {base}")

            rec = parse_ascii_ply(ply_path)
            pred, xyz_raw = infer_scene_chunked(model, rec, args, device, expected_in, scene_name=base)

            # Export per-class PLYs
            for c in range(args.num_classes):
                mask = (pred == c)
                if not np.any(mask):
                    continue
                xyz_c = xyz_raw[mask]
                color_c = np.repeat(PALETTE[c][None, :], xyz_c.shape[0], axis=0)
                out_ply_c = os.path.join(out_scene_dir, f"{base}_class{c}.ply")
                write_ply_xyzrgb(out_ply_c, xyz_c, color_c)

            # Per-scene metrics row
            counts = [int(np.sum(pred == c)) for c in range(args.num_classes)]
            if ('label' in rec) and (rec['label'] is not None):
                gt = rec['label'].astype(np.int64)
                cm = confusion_matrix(args.num_classes, gt, pred)
                cm_total += cm
                scores = scores_from_cm(cm)
                row = [base, int(len(pred))] + counts + \
                      [scores["OA"], scores["mAcc"], scores["mIoU"]] + \
                      [float(x) for x in scores["IoU_per_class"]] + \
                      [float(x) for x in scores["Acc_per_class"]]
            else:
                row = [base, int(len(pred))] + counts + ["NA","NA","NA"] + ["NA"]*(2*args.num_classes)
            writer.writerow(row)

    # Dataset-level metrics
    global_csv = os.path.join(args.outdir, "dataset_metrics.csv")
    if cm_total.sum() > 0:
        scores_glob = scores_from_cm(cm_total)
        with open(global_csv, "w", newline="", encoding="utf-8") as gf:
            gw = csv.writer(gf)
            gw.writerow(["metric", "value"])
            gw.writerow(["OA",   scores_glob["OA"]])
            gw.writerow(["mAcc", scores_glob["mAcc"]])
            gw.writerow(["mIoU", scores_glob["mIoU"]])
            for c in range(args.num_classes):
                name_c = class_names[c] if c < len(class_names) else f"class_{c}"
                gw.writerow([f"IoU_{name_c}", float(scores_glob["IoU_per_class"][c])])
            for c in range(args.num_classes):
                name_c = class_names[c] if c < len(class_names) else f"class_{c}"
                gw.writerow([f"Acc_{name_c}", float(scores_glob["Acc_per_class"][c])])

        print("\n=== Dataset-level metrics ===")
        for k, v in scores_glob.items():
            if isinstance(v, (list, np.ndarray)):
                print(f"{k}: {[round(float(x), 4) for x in v]}")
            else:
                print(f"{k}: {float(v):.4f}")
    else:
        print("\n(No ground-truth labels found; only per-class PLYs were exported.)")

if __name__ == "__main__":
    main()
