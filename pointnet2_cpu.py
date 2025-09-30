# pointnet2_cpu.py
# Minimal PointNet++ (SSG) for semantic segmentation — pure PyTorch, CPU-friendly
# Implements: FPS, Ball Query (radius), Set Abstraction (SSG),
# Feature Propagation (3-NN interpolation), Seg head.
# No custom CUDA ops required.

from typing import List, Optional, Tuple
import torch
import torch.nn as nn


# ----------------------------
# Sampling & Neighborhood Ops
# ----------------------------

def fps(x: torch.Tensor, m: int) -> torch.Tensor:
    """Farthest Point Sampling (naive CPU version)."""
    B, N, _ = x.shape
    device = x.device
    idx = torch.zeros(B, m, dtype=torch.long, device=device)
    farthest = torch.randint(0, N, (B,), device=device)
    dist = torch.full((B, N), 1e10, device=device)
    batch_indices = torch.arange(B, device=device)
    for i in range(m):
        idx[:, i] = farthest
        centroid = x[batch_indices, farthest, :].unsqueeze(1)     # (B,1,3)
        d = torch.sum((x - centroid) ** 2, dim=2)                 # (B,N)
        dist = torch.minimum(dist, d)
        farthest = torch.max(dist, dim=1).indices
    return idx


def ball_query(x: torch.Tensor, centroids: torch.Tensor, radius: float, K: int) -> torch.Tensor:
    """Radius-based neighborhood with top-K closest inside radius."""
    dists = torch.cdist(centroids, x, p=2)        # (B, M, N)
    within = dists <= radius
    dists_masked = dists.clone()
    dists_masked[~within] = 1e10
    idx = torch.topk(dists_masked, k=K, dim=2, largest=False).indices  # (B,M,K)
    return idx


def index_points(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points/features by index.
    x: (B, N, C), idx: (B, M, K) → (B, M, K, C)
    """
    B, N, C = x.shape
    batch_indices = torch.arange(B, device=x.device).view(B, 1, 1).expand(-1, idx.size(1), idx.size(2))
    return x[batch_indices, idx, :]


# ----------------------------
# Interpolation (Feature Prop)
# ----------------------------

def three_nn_interpolate(xyz_src: torch.Tensor,
                         xyz_dst: torch.Tensor,
                         feat_dst: torch.Tensor) -> torch.Tensor:
    """Interpolate features from (xyz_dst, feat_dst) onto xyz_src using inverse-distance 3-NN."""
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


# ----------------------------
# Set Abstraction (SSG)
# ----------------------------

class SA_SSG(nn.Module):
    """Single-Scale Grouping SA layer."""
    def __init__(self, npoint: int, radius: float, nsample: int,
                 in_channels: int, mlp: List[int]) -> None:
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

    def forward(self, xyz: torch.Tensor, features: Optional[torch.Tensor]):
        B, N, _ = xyz.shape
        idx = fps(xyz, self.npoint)                                    # (B, npoint)
        new_xyz = xyz[torch.arange(B).unsqueeze(-1), idx]              # (B, npoint, 3)
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


# ----------------------------
# Feature Propagation (FP)
# ----------------------------

class FP(nn.Module):
    def __init__(self, in_channels: int, mlp: List[int]) -> None:
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


# ----------------------------
# PointNet++ (SSG) Segmentation
# ----------------------------

class PointNet2_SSG_Seg(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 4) -> None:
        super().__init__()
        #On CPU it takes much long time, 1 epoch takes around 2 hours!
        # self.sa1 = SA_SSG(1024, 0.10, 32, in_channels, [64, 64, 128])
        # self.sa2 = SA_SSG(256,  0.20, 32, 128,        [128, 128, 256])
        # self.sa3 = SA_SSG(64,   0.40, 32, 256,        [256, 256, 512])

        #Tweeking for faster response by reducing the parameter values
        self.sa1 = SA_SSG(512,  0.10, 16, in_channels, [64, 64, 128])  # was 1024, nsample 32
        self.sa2 = SA_SSG(128,  0.20, 16, 128,         [128, 128, 256]) # was 256
        self.sa3 = SA_SSG(32,   0.40, 16, 256,         [256, 256, 512]) # was 64

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

        skip0 = feats if feats is not None else None
        up0 = self.fp1(xyz, l1_xyz, skip0, l1_up)

        return self.head(up0)   # (B,num_classes,N)
