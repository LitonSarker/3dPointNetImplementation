# infer_pointnet2_cpu.py
# Self-contained, CPU-only PointNet++ (SSG) inference (no external model imports). Inline implementation.
# Robust to checkpoint channels (auto-infers from weights). Exports PLY + CSV (+ optional labels.txt).

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional

# ============ Core ops ============
# Farthest Point Sampling (FPS), Ball Query, Indexing, 3-NN Interpolation
# Farthest Point Sampling): Selects m points from the input cloud by iteratively picking the farthest points to ensure good spatial coverage
 
def fps(x: torch.Tensor, m: int) -> torch.Tensor:
    B, N, _ = x.shape
    device = x.device
    idx = torch.zeros(B, m, dtype=torch.long, device=device)
    farthest = torch.randint(0, N, (B,), device=device)         # Initial farthest point indices
    dist = torch.full((B, N), 1e10, device=device)              # Initialize distances to large values
    batch_indices = torch.arange(B, device=device)              # Batch indices for advanced indexing
    for i in range(m):
        idx[:, i] = farthest
        centroid = x[batch_indices, farthest, :].unsqueeze(1)   # (B,1,3)
        d = torch.sum((x - centroid) ** 2, dim=2)               # (B,N) squared distances to the new centroid
        dist = torch.minimum(dist, d)                           # Update minimum distances
        farthest = torch.max(dist, dim=1).indices               # Next farthest point
    return idx

# Ball Query: For each centroid, finds up to K nearest neighbors within a specified radius, returning their indices
# For each centroid, finds up to K neighboring points within a given radius in the point cloud
# If fewer than K points are found, the nearest points are duplicated to ensure K neighbors
def ball_query(x: torch.Tensor, centroids: torch.Tensor, radius: float, K: int) -> torch.Tensor:
    dists = torch.cdist(centroids, x, p=2)          # (B,M,N) pairwise distances
    within = dists <= radius                        # (B,M,N) boolean mask of points within radius
    dists_masked = dists.clone()                    # Mask distances outside radius
    dists_masked[~within] = 1e10                    # Large value for points outside radius
    idx = torch.topk(dists_masked, k=K, dim=2, largest=False).indices       # (B,M,K)
    return idx

# Indexing: Gathers points or features based on provided indices
# Gathers points or features from the input tensor based on the provided indices
def index_points(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    B, N, C = x.shape           # (B,N,C)
    batch_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(-1, idx.size(1), idx.size(2))     # (B,M,K)
    return x[batch_idx, idx, :]         # (B,M,K,C)

#Interpolates features for source points by weighting the 3 nearest neighbors from destination points based on inverse distance.
# Uses torch.cdist for distance computation and gathers features accordingly.
def three_nn_interpolate(xyz_src: torch.Tensor, xyz_dst: torch.Tensor, feat_dst: torch.Tensor) -> torch.Tensor:
    d = torch.cdist(xyz_src, xyz_dst, p=2) + 1e-8               # (B,Ns,Nd) pairwise distances
    idx = torch.topk(d, k=3, dim=2, largest=False).indices      # (B,Ns,3)
    d3 = torch.gather(d, 2, idx)                                # (B,Ns,3)
    w = (1.0 / d3)
    w = w / torch.sum(w, dim=2, keepdim=True)
    w = w.unsqueeze(1)
    feat_dst_perm = feat_dst.permute(0, 2, 1)
    gathered = index_points(feat_dst_perm, idx)           # (B,Ns,3,Cd)
    gathered = gathered.permute(0, 3, 1, 2).contiguous()  # (B,Cd,Ns,3)
    return torch.sum(gathered * w, dim=3)                 # (B,Cd,Ns)

# ============ Blocks ============
# Set Abstraction (SA) with Single-Scale Grouping (SSG) and Feature Propagation (FP)
# Each SA layer downsamples points and extracts local features using MLPs and max pooling.
# Each FP layer upsamples and refines features by interpolating from a coarser level

class SA_SSG(nn.Module):
    def __init__(self, npoint: int, radius: float, nsample: int, in_channels: int, mlp: List[int]):
        super().__init__()
        self.npoint = npoint; self.radius = radius; self.nsample = nsample
        layers = []; last_c = in_channels + 3       # in_channels + 3 (relative xyz)
        for out_c in mlp:                           # +3 for relative xyz
            layers += [nn.Conv2d(last_c, out_c, 1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)]       # 1x1 conv, BN, ReLU
            last_c = out_c
        self.mlp = nn.Sequential(*layers)       # Sequential MLP

    # Forward pass: samples points, groups neighbors, applies MLP, and pools features
    def forward(self, xyz: torch.Tensor, features: Optional[torch.Tensor]):
        B, N, _ = xyz.shape
        idx = fps(xyz, self.npoint)
        new_xyz = xyz[torch.arange(B).unsqueeze(-1), idx]
        knn_idx = ball_query(xyz, new_xyz, self.radius, self.nsample)       # (B,npoint,nsample)
        grouped_xyz = index_points(xyz, knn_idx)                            # (B,npoint,nsample,3)
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)               # Normalize
        if features is not None:
            feat = features.permute(0, 2, 1).contiguous()
            grouped_feat = index_points(feat, knn_idx)
            new_feat = torch.cat([grouped_xyz_norm, grouped_feat], dim=-1)
        else:
            new_feat = grouped_xyz_norm
        new_feat = new_feat.permute(0, 3, 1, 2).contiguous()                # (B,C,npoint,nsample)
        new_feat = self.mlp(new_feat)                                       # (B,C',npoint,nsample) 
        new_feat = torch.max(new_feat, dim=3).values
        return new_xyz, new_feat

# Feature Propagation (FP) layer: interpolates and refines features from a coarser level to a finer level
class FP(nn.Module):
    def __init__(self, in_channels: int, mlp: List[int]):
        super().__init__()
        layers = []; last_c = in_channels
        for out_c in mlp:
            layers += [nn.Conv1d(last_c, out_c, 1), nn.BatchNorm1d(out_c), nn.ReLU(inplace=True)]
            last_c = out_c
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz_src, xyz_dst, feat_src, feat_dst):
        interpolated = three_nn_interpolate(xyz_src, xyz_dst, feat_dst)
        fused = torch.cat([interpolated, feat_src], dim=1) if feat_src is not None else interpolated
        return self.mlp(fused)

# ============ Model ============
# PointNet2 with Single-Scale Grouping (SSG) for point cloud segmentation
# The model consists of three Set Abstraction (SA) layers followed by three Feature Propagation (FP) layers and a final classification head
# Inline implementation for CPU inference, from original PointNet++ architecture

class PointNet2_SSG_Seg(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        # Lightweight CPU config
        self.sa1 = SA_SSG(512, 0.10, 16, in_channels, [64, 64, 128])        # +3 for relative xyz
        self.sa2 = SA_SSG(128, 0.20, 16, 128,         [128, 128, 256])      # +3 for relative xyz
        self.sa3 = SA_SSG(32,  0.40, 16, 256,         [256, 256, 512])      # +3 for relative xyz
        self.fp3 = FP(512 + 256, [256, 256])                                # +256 from skip connection
        self.fp2 = FP(256 + 128, [256, 128])                                # +128 from skip connection     
        self.fp1 = FP(128 + in_channels, [128, 128, 128])                   # +in_channels from input features  
        
        # Final classification head, 1x1 conv, BN, ReLU, Dropout, 1x1 conv, output, logits,
        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),         # 1x1 conv, BN, ReLU
            nn.Dropout(0.1),                                                            # Dropout
            nn.Conv1d(128, num_classes, 1)                                              # 1x1 conv to num_classes     
        )

    # Forward pass through SA and FP layers, followed by the classification head
    # xyz: (B,N,3), feats: (B,C,N) or None, Returns: logits (B,num_classes,N)

    def forward(self, xyz: torch.Tensor, feats: Optional[torch.Tensor] = None):
        l1_xyz, l1_feat = self.sa1(xyz, feats)
        l2_xyz, l2_feat = self.sa2(l1_xyz, l1_feat)
        l3_xyz, l3_feat = self.sa3(l2_xyz, l2_feat)
        l2_up = self.fp3(l2_xyz, l3_xyz, l2_feat, l3_feat)
        l1_up = self.fp2(l1_xyz, l2_xyz, l1_feat, l2_up)
        up0 = self.fp1(xyz, l1_xyz, feats, l1_up)
        return self.head(up0)

# ============ IO helpers ============
def _try_import_open3d():
    try:
        import open3d as o3d
        return o3d
    except Exception:
        return None
# Read PLY, return (N,3) xyz, (N,3) rgb or None, (N,3) normals or None
# Supports Open3D or plyfile, prefers Open3D if available, raises if neither is installed
def _read_ply_xyz_rgb_n(o3d, ply_path):
    ply_path = str(ply_path)
    if o3d is not None:
        pcd = o3d.io.read_point_cloud(ply_path)
        if pcd.is_empty(): raise ValueError(f"Open3D read empty point cloud from {ply_path}")
        xyz = np.asarray(pcd.points, dtype=np.float32)
        rgb = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None
        normals = np.asarray(pcd.normals, dtype=np.float32) if pcd.has_normals() else None
        return xyz, rgb, normals
    try:
        from plyfile import PlyData
        pd = PlyData.read(ply_path); el = pd["vertex"]
        xyz = np.stack([el["x"], el["y"], el["z"]], axis=1).astype(np.float32)
        rgb = None; normals = None
        if all(k in el.data.dtype.names for k in ("red","green","blue")):
            rgb = np.stack([el["red"], el["green"], el["blue"]], axis=1).astype(np.float32)/255.0
        if all(k in el.data.dtype.names for k in ("nx","ny","nz")):
            normals = np.stack([el["nx"], el["ny"], el["nz"]], axis=1).astype(np.float32)
        return xyz, rgb, normals
    except Exception as e:
        raise RuntimeError("Install open3d or plyfile to read PLY.") from e

# Write PLY with colors (N,3) xyz, (N,3) colors in [0,1], using Open3D or plyfile, prefers Open3D if available
# Raises if neither is installed

def _write_ply_with_colors(o3d, out_path, xyz, colors):
    if o3d is None:
        try:
            from plyfile import PlyData, PlyElement
            pts = np.concatenate([xyz, (colors * 255).astype(np.uint8)], axis=1)
            dtype = [("x","f4"),("y","f4"),("z","f4"),("red","u1"),("green","u1"),("blue","u1")]
            arr = np.array([tuple(row) for row in pts], dtype=dtype)
            PlyData([PlyElement.describe(arr, "vertex")], text=True).write(out_path); return
        except Exception as e:
            raise RuntimeError("Install open3d or plyfile to write PLY.") from e
    import open3d as o3d_local
    pcd = o3d_local.geometry.PointCloud()
    pcd.points = o3d_local.utility.Vector3dVector(xyz)
    pcd.colors = o3d_local.utility.Vector3dVector(colors.clip(0,1))
    o3d_local.io.write_point_cloud(out_path, pcd, write_ascii=True)

# ============ Utils ============
# Tries to detect the input channel size (in_channels) from a checkpoint dictionary.
def _infer_in_channels_from_ckpt(ckpt):
    for key in ("hparams","config","model_args"):                       
        if isinstance(ckpt.get(key), dict):
            for k2 in ("in_channels","input_channels"):
                v = ckpt[key].get(k2)
                if isinstance(v, (int, float)): return int(v)
    for k in ("in_channels","input_channels"):
        v = ckpt.get(k)
        if isinstance(v, (int, float)): return int(v)
    return None

# Extracts the actual model weights (state_dict) from a checkpoint container.
def _unwrap_state_dict(ckpt):
    for k in ("model_state","state_dict","model"):
        if isinstance(ckpt.get(k), dict): return ckpt[k]
    if isinstance(ckpt, dict): return ckpt
    raise KeyError("No state_dict found in checkpoint.")

# Concatenates selected features (XYZ, RGB, normals) into a single input array.
def _build_features(xyz, rgb, normals, use_rgb, use_normals):
    feats = [xyz]; order = ["XYZ"]
    if use_rgb and (rgb is not None): feats.append(rgb); order.append("RGB")
    if use_normals and (normals is not None): feats.append(normals); order.append("normals")
    X = np.concatenate(feats, axis=1).astype(np.float32)
    return X, order

# Adjusts feature dimensions to match the expected channel size by trimming or zero-padding
def _truncate_or_pad(X, want_c, order):
    have = X.shape[1]
    if (want_c is None) or (want_c == have): return X
    if have > want_c:
        print(f"[INFO] Truncating {have}->{want_c} (order={','.join(order)})")
        return X[:, :want_c]
    print(f"[INFO] Padding {have}->{want_c} (order={','.join(order)})")
    pad = np.zeros((X.shape[0], want_c - have), dtype=X.dtype)
    return np.concatenate([X, pad], axis=1)

# Creates a fixed random color palette for visualizing class labels.
def _label_palette(n):
    rng = np.random.default_rng(13)
    cols = rng.random((n, 3)); cols[0] = np.array([0.2,0.2,0.2]); return cols

# ============ Main ============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--num_classes", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_ply", required=True)
    ap.add_argument("--save_labels", action="store_true")
    ap.add_argument("--use_rgb", action="store_true")
    ap.add_argument("--use_normals", action="store_true")
    ap.add_argument("--class_names", type=str,
                    default="ceiling,floor,wall,beam,column,window,door,table,chair,sofa,bookcase,board,clutter")
    args = ap.parse_args()

    # Read input
    o3d = _try_import_open3d()
    xyz_np, rgb_np, normals_np = _read_ply_xyz_rgb_n(o3d, args.ply)

    # Features from file
    X, order = _build_features(xyz_np, rgb_np, normals_np, args.use_rgb, args.use_normals)

    # Load ckpt + unwrap + infer channels from WEIGHTS
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = _unwrap_state_dict(ckpt)

    # Determines how many input channels the checkpointed model expects and reshapes the feature matrix accordingly so it can be fed into the network without mismatch errors.
    sa1_key = None
    for cand in ("sa1.mlp.0.weight","module.sa1.mlp.0.weight"):         
        if any(k.endswith(cand) for k in sd.keys()):
            sa1_key = next(k for k in sd.keys() if k.endswith(cand)); break
    if sa1_key is None:
        feat_meta = _infer_in_channels_from_ckpt(ckpt)
        sa1_in_total = (feat_meta + 3) if (feat_meta is not None) else X.shape[1]
    else:
        sa1_in_total = sd[sa1_key].shape[1]

    feat_ch = max(0, sa1_in_total - 3)   # features expected by ckpt
    X = _truncate_or_pad(X, sa1_in_total, order)

    # Split tensors
    xyz = torch.from_numpy(X[:, :3]).float().unsqueeze(0)     # (1,N,3)
    feats = None
    if feat_ch > 0:                              # Take extra channels (e.g., RGB/normals), transpose to (C, N), add batch dim → (1, C, N)
        feats_np = X[:, 3:3+feat_ch].T[None, ...].astype(np.float32)  # (1,C,N)
        feats = torch.from_numpy(feats_np).float()

    # num_classes from head if present, # --- Determine number of classes from checkpoint head (if available) ---

    head_key = None
    for cand in ("head.4.weight","module.head.4.weight"):
        if any(k.endswith(cand) for k in sd.keys()):
            head_key = next(k for k in sd.keys() if k.endswith(cand)); break
    num_classes = sd[head_key].shape[0] if head_key is not None else int(args.num_classes)

    # Build Model &  load weights, set to eval on CPU ---

    model = PointNet2_SSG_Seg(in_channels=feat_ch, num_classes=num_classes)
    model.load_state_dict(sd, strict=False)
    model.eval().to("cpu")

    # Inference,  Run inference: forward pass only, no gradient ---

    with torch.no_grad():
        logits = model(xyz, feats)                      # (1,num_classes,N)
        preds = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)

    # --- Colorize predictions and save as PLY ---
    palette = _label_palette(num_classes)
    colors = palette[preds]
    out_dir = Path(args.out_ply).parent; out_dir.mkdir(parents=True, exist_ok=True)
    _write_ply_with_colors(o3d, args.out_ply, xyz_np, colors)

    # --- Save labels as .txt (optional) ---

    if args.save_labels:
        labels_path = str(Path(args.out_ply).with_suffix("")) + "_labels.txt"
        np.savetxt(labels_path, preds, fmt="%d")
        print(f"[OK] Saved labels: {labels_path}")

    # Save CSV (x,y,z,label,class_name)
    names = [s.strip() for s in args.class_names.split(",")]
    if len(names) < num_classes:
        names += [f"class_{i}" for i in range(len(names), num_classes)]
    csv_path = str(Path(args.out_ply).with_suffix("")) + "_labels.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("x,y,z,label,class_name\n")
        for (x, y, z), lab in zip(xyz_np, preds):
            cname = names[int(lab)] if 0 <= int(lab) < len(names) else f"class_{int(lab)}"
            f.write(f"{x},{y},{z},{int(lab)},{cname}\n")
    print(f"[OK] Saved CSV: {csv_path}")

    print(f"[OK] Wrote colorized predictions: {args.out_ply}")

if __name__ == "__main__":
    main()
