# 3D PointNet++ for Construction Progress Monitoring (CPU-Only)

This repository implements **PointNet++** for semantic segmentation of 3D point clouds in the context of **construction progress monitoring**.  
It aims to automate monitoring when up-to-date BIM models are not available, by using real-time point cloud data (from LiDAR, CCTV, drones, etc.) as a **pseudo-BIM** substitute.

---

## 📌 Project Context

Construction projects often lack reliable and updated BIM models, which makes manual progress tracking costly and inefficient.  
3D point clouds provide an alternative by capturing as-built conditions. This project demonstrates an **automation pipeline using PointNet++** to segment building elements from point cloud data, supporting real-time progress monitoring.  

Target audience: **researchers, industry engineers, and PhD advisors** interested in computer vision and construction management.

---

## ✨ Features

- CPU-only implementation of **PointNet++ (SSG)** for semantic segmentation.  
- Supports **XYZ** or **XYZ+RGB** features.  
- Automatic **train/validation split (80:20)** from raw S3DIS dataset.  
- Tracks progress via **OA, mAcc, mIoU** metrics.  
- Checkpointing (`last_model.pth`, `best_model.pth`) and logging (`history.json`, `train_log.json`).  
- Preprocessing scripts to convert **S3DIS annotations → PLY files**.  

---

## 📂 Dataset Information

- Dataset: [Stanford Large-Scale 3D Indoor Spaces (S3DIS)](https://cvg-data.inf.ethz.ch/s3dis/)  
- Format: RGB-D point clouds → converted to `.ply`.  

### 1. Convert annotations to `.ply`
```bash
python s3dis_annots_to_ply_general.py   --root /path/to/Stanford3dDataset_v1.2_Aligned_Version   --dst  /path/to/out_ply
```

### 2. Create train/val splits (80:20)
```bash
python make_splits_3dis.py --src_dir .\out_ply --val_ratio 0.2
```

---

## ⚙️ Requirements

### Python environment
```bash
# create venv
py -3.12 -m venv .venv
.\.venv\Scriptsctivate

# upgrade pip
python -m pip install --upgrade pip
```

### Install dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install open3d==0.19.0
pip install "git+https://github.com/isl-org/Open3D-ML@v0.18.0"
pip install numpy scipy pandas matplotlib plyfile
```

> This implementation is **CPU-only**.

---

## 🔄 Pipeline Overview

![Pipeline](assets/Process_Flow.png)

## 🚀 Usage

### Training
```bash
python train_eval_pointnet2_cpu.py   --train_files .\out_ply	rain_med.txt   --val_files   .\out_plyal_med.txt   --num_classes 13   --epochs 40   --batch_size 1   --num_points 512   --outdir runs\seg_med
```

### Outputs
- Checkpoints: `runs/seg_cpu/checkpoints/{last_model.pth, best_model.pth}`  
- Metrics history: `runs/seg_cpu/history.json`  
- Training logs: `runs/seg_cpu/train_log.json`  

---

## 📁 Folder Structure

```
├── s3dis_annots_to_ply_general.py    # Convert S3DIS txt → PLY
├── make_splits_3dis.py               # Train/val split (80:20)
├── train_eval_pointnet2_cpu.py       # Training + evaluation
├── infer_pointnet2_cpu.py            # Inference on new data
├── eval_list_single.py               # Eval on single list
├── export_class_views_cpu.py         # Export class views
├── occlusion_eval_cpu.py             # Occlusion evaluation
├── .venv/                            # Virtual environment
├── class_views/                      # Subset views (hallway, etc.)
├── out_ply/                          # Processed PLYs by area
├── Pointnet2_PyTorch/                # PointNet++ implementation
├── results/                          # Predictions + CSV exports
└── runs/
    └── seg_cpu/
        ├── checkpoints/
        │   ├── best_model.pth
        │   └── last_model.pth
        ├── history.json
        └── train_log.json
```

---

## 📊 Results

- Training loss decreased **1.92 → 1.30** across 12 epochs.  
- Training metrics improved: **OA 0.425 / mIoU 0.092 → OA 0.625 / mIoU 0.251**.  
- Validation metrics (evaluated every 2 epochs):  
  - mIoU: 0.072 (ep2) → 0.229 (ep12)  
  - OA: 0.599 (ep12)  

---

## 🏗️ Applications in Construction

- Real-time **progress monitoring** from 3D point clouds.  
- Comparison of **as-built vs BIM models** (or pseudo-BIM).  
- Detecting **installed vs missing elements** on site.  
- Compatible with data from **LiDAR, CCTV, drones**.  

---

## 🔮 Next Steps / TODO

- Multi-modal fusion (point cloud + CCTV/drone imagery).  
- Larger-scale datasets.  
- GPU support for faster training.  
- Explainability (Grad-CAM, attention maps).  

---

## 📚 Citation & Credits

- PointNet++: [Qi et al. (2017)](https://arxiv.org/abs/1706.02413)  
- Implementation support: **GPT-5.0**
