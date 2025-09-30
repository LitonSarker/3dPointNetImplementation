import os
import numpy as np
import torch
from torch.utils.data import Dataset
import open3d as o3d

def _parse_ascii_ply(path):
    """
    Parses ASCII .ply with header containing 'element vertex N' and properties.
    Expected vertex line supports: x y z [red green blue] [label]
    Returns: dict with keys: 'xyz' (N,3), 'rgb' (N,3 or None), 'label' (N,) or None
    """
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        header = []
        line = f.readline()
        if not line.startswith('ply'):                  #PLY file format check
            raise ValueError('Not a PLY file')
        header.append(line)
        num_vertices = None
        properties = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError('Unexpected EOF in header')
            header.append(line)
            if line.startswith('element vertex'):
                num_vertices = int(line.strip().split()[-1])        #Number of vertices
            if line.startswith('property'):
                parts = line.strip().split()
                if len(parts) >= 3:
                    properties.append(parts[-1])                    #Property name that covers x,y,z,r,g,b,label (classes definition)
            if line.strip() == 'end_header':
                break
        data = []
        for i in range(num_vertices):                 #Read vertex lines/points
            ln = f.readline()
            if not ln:
                break
            parts = ln.strip().split()
            if not parts:
                continue
            data.append(parts)                              #List of lists of strings
        arr = np.array(data, dtype=np.float32)              #Convert to numpy array of floats
        if arr.shape[1] < 3:
            pcd = o3d.io.read_point_cloud(path)             #Fallback: use Open3D if less than 3 properties (i,e if there's no x,y,z)
            xyz = np.asarray(pcd.points, dtype=np.float32)  #(N,3), xyz coordinates are stored here
            rgb = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None    #RGB values are stored here
            return {'xyz': xyz, 'rgb': rgb, 'label': None}  #returns coordinates and colors if available
        name_to_col = {name: idx for idx, name in enumerate(properties[:arr.shape[1]])}     #Map property names to column indices
        xyz_cols = [name_to_col.get('x', 0), name_to_col.get('y', 1), name_to_col.get('z', 2)]  #Get indices of x,y,z columns
        xyz = arr[:, xyz_cols].astype(np.float32)
        r_name = 'red' if 'red' in name_to_col else ('r' if 'r' in name_to_col else None)
        g_name = 'green' if 'green' in name_to_col else ('g' if 'g' in name_to_col else None)
        b_name = 'blue' if 'blue' in name_to_col else ('b' if 'b' in name_to_col else None)
        if r_name and g_name and b_name:
            rgb = arr[:, [name_to_col[r_name], name_to_col[g_name], name_to_col[b_name]]].astype(np.float32)
            if rgb.max() > 1.0:             #If RGB values are in 0-255 range, convert to 0-1
                rgb = rgb / 255.0           #Normalize to 0..1 for more generalization
        else:
            pcd = o3d.io.read_point_cloud(path)        #Fallback: use Open3D if RGB columns are not found
            rgb = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None    #RGB values are stored here
        lbl = None
        for cand in ['label', 'seg_label', 'class', 'y']:   #Check for possible label column names
            if cand in name_to_col:
                col = name_to_col[cand]
                lbl = arr[:, col].astype(np.int64)
                break
        return {'xyz': xyz, 'rgb': rgb, 'label': lbl}

class PLYSegDataset(Dataset):
    def __init__(self, list_file, num_points=4096, use_rgb=False, augment=True):
        with open(list_file, "r", encoding="utf-8-sig") as f:  # handles UTF-8 with BOM
            self.files = [ln.strip().lstrip("\ufeff") for ln in f if ln.strip()]
        self.num_points = num_points
        self.use_rgb = use_rgb
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def _augment(self, xyz):
        xyz = xyz.copy()
        xyz += np.random.normal(scale=0.005, size=xyz.shape).astype(np.float32)
        theta = np.random.uniform(-np.pi/18, np.pi/18)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)
        xyz = (xyz @ R.T).astype(np.float32)
        return xyz

    def __getitem__(self, idx):
        path = self.files[idx]
        rec = _parse_ascii_ply(path)
        xyz = rec['xyz'].astype(np.float32)
        rgb = rec['rgb']
        lbl = rec['label']
        if lbl is None:
            raise ValueError(f'No labels found in {path}. Expected a vertex "label" property.')
        N = xyz.shape[0]
        choice = np.random.choice(N, self.num_points, replace=(N < self.num_points))
        xyz = xyz[choice]
        lbl = lbl[choice]
        if rgb is not None and self.use_rgb:
            rgb = rgb[choice]
        if self.augment:
            xyz = self._augment(xyz)
        if self.use_rgb and rgb is not None:
            feats = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
        else:
            feats = xyz.astype(np.float32)
        xyz_t = torch.from_numpy(xyz).float()
        feats_t = torch.from_numpy(feats).float().permute(1,0)
        lbl_t = torch.from_numpy(lbl).long()
        return xyz_t, feats_t, lbl_t, os.path.basename(path)
