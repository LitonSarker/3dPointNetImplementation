# train_eval_pointnet2_cpu.py
# CPU-only training/evaluation for PointNet++ (SSG) semantic segmentation
# - Keeps PointNet++ implementation: from pointnet2_cpu import PointNet2_SSG_Seg
# - Adds robust logging, sanity checks, batch-level try/except, JSON logs
# - Ensures all data is considered (drop_last=False)
# - Provides fast-debug flags to finish quickly on CPU

import argparse, os, sys, time, json, random
import numpy as np
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.utils.data import DataLoader  # type: ignore

from dataset_ply_cpu import PLYSegDataset
from pointnet2_cpu import PointNet2_SSG_Seg
from metrics_pointcloud_cpu import confusion_matrix, scores_from_cm

CLASS_NAMES = [
    "ceiling","floor","wall","beam","column","window",
    "door","table","chair","sofa","bookcase","board","clutter"
]

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def configure_logging():
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("train")


@torch.no_grad()
def evaluate(model, loader, device, num_classes, log_interval=0):
    model.eval()
    ce = nn.CrossEntropyLoss()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_loss = 0.0
    seen = 0

    for bidx, batch in enumerate(loader, start=1):
        xyz, feats, labels, _ = batch
        xyz = xyz.to(device, non_blocking=False)
        feats = feats.to(device, non_blocking=False)
        labels = labels.to(device, non_blocking=False)

        logits = model(xyz, feats)
        loss = ce(logits, labels)

        bs = xyz.size(0)
        total_loss += float(loss.item()) * bs
        seen += bs

        pred = torch.argmax(logits, dim=1)
        cm += confusion_matrix(
            num_classes,
            labels.cpu().numpy().ravel(),
            pred.cpu().numpy().ravel(),
        )

        if log_interval and (bidx % log_interval == 0):
            print(json.dumps({"phase": "eval", "step": bidx, "loss": float(loss.item())}), flush=True)

    avg_loss = total_loss / max(1, seen)
    scores = scores_from_cm(cm)
    return avg_loss, scores


def main():
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    ap = argparse.ArgumentParser(description="CPU-Only PointNet++ (SSG) train/eval")
    # Required IO
    ap.add_argument("--train_files", type=str, required=True, help="txt file: one PLY path per line")
    ap.add_argument("--val_files",   type=str, required=True, help="txt file: one PLY path per line")
    ap.add_argument("--num_classes", type=int, required=True)

    # Core hyperparams
    ap.add_argument("--epochs",      type=int, default=50)
    ap.add_argument("--batch_size",  type=int, default=4)
    ap.add_argument("--num_points",  type=int, default=4096)
    ap.add_argument("--use_rgb",     action="store_true")
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    # System / IO
    ap.add_argument("--outdir",      type=str, default="runs/seg_cpu")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--chunk_size",  type=int, default=80000, help="points per chunk for inference (CPU memory safety)")

    # Quality of life
    ap.add_argument("--log_interval",        type=int, default=10, help="print every N steps")
    ap.add_argument("--max_steps_per_epoch", type=int, default=0,  help="debug: limit train steps per epoch (0 = all)")
    ap.add_argument("--eval_every",          type=int, default=1,  help="run eval every N epochs")
    ap.add_argument("--seed",                type=int, default=42)

    args = ap.parse_args()

    # Logging, seed, threads
    log = configure_logging()
    log.info(">>> Script started (CPU-only)")
    set_seed(args.seed)
    try:
        # Respect OMP_NUM_THREADS if set; otherwise default to 6 to avoid oversubscription on CPU
        torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "6"))))
    except Exception:
        pass

    # Output dirs
    os.makedirs(args.outdir, exist_ok=True)
    ckpt_dir = os.path.join(args.outdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device("cpu")

    # Datasets
    train_ds = PLYSegDataset(args.train_files, num_points=args.num_points, use_rgb=args.use_rgb, augment=True)
    val_ds   = PLYSegDataset(args.val_files,   num_points=args.num_points, use_rgb=args.use_rgb, augment=False)

    # DataLoaders (Windows/CPU safe) — do not drop samples
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,  # keep 0 on Windows to avoid hangs
        pin_memory=False,
        drop_last=False,               # ensure every sample is used
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False,
    )

    log.info(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)} | "
             f"Batch size: {args.batch_size} | Num points: {args.num_points}")

    # Fail-fast sanity check (surface dataset/transform/path bugs immediately)
    try:
        sb = next(iter(train_loader))
        shapes = [getattr(t, "shape", None) for t in (sb if isinstance(sb, (list, tuple)) else [sb])]
        log.info(f"Sanity batch OK. Shapes: {shapes}")
    except Exception as e:
        log.error(f"Sanity batch FAILED: {e}", exc_info=True)
        return

    # Model (KEEPING your PointNet++ implementation)
    in_ch = 6 if args.use_rgb else 3
    model = PointNet2_SSG_Seg(in_channels=in_ch, num_classes=args.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # after optimizer
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_miou = -1.0
    history = []

    try:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()

            # Train accumulators
            cm_tr = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
            total_loss_tr = 0.0
            seen_tr = 0

            for bidx, batch in enumerate(train_loader, start=1):
                try:
                    xyz, feats, labels, _ = batch
                    xyz = xyz.to(device, non_blocking=False)
                    feats = feats.to(device, non_blocking=False)
                    labels = labels.to(device, non_blocking=False)

                    optimizer.zero_grad()
                    logits = model(xyz, feats)
                    loss = ce(logits, labels)
                    loss.backward()
                    optimizer.step()

                    bs = xyz.size(0)
                    total_loss_tr += float(loss.item()) * bs
                    seen_tr += bs

                    with torch.no_grad():
                        pred = torch.argmax(logits, dim=1)
                        cm_tr += confusion_matrix(
                            args.num_classes,
                            labels.cpu().numpy().ravel(),
                            pred.cpu().numpy().ravel(),
                        )

                    if args.log_interval and (bidx % args.log_interval == 0):
                        print(json.dumps({"phase": "train", "epoch": epoch, "step": bidx, "loss": float(loss.item())}), flush=True)

                    if args.max_steps_per_epoch and bidx >= args.max_steps_per_epoch:
                        break  # fast debug exit

                except Exception as e:
                    # Skip only the offending batch; keep training going
                    print(json.dumps({"phase": "train", "epoch": epoch, "step": bidx, "error": str(e)}), flush=True)
                    continue

            tr_loss = total_loss_tr / max(1, seen_tr)
            tr_scores = scores_from_cm(cm_tr)

            # Evaluate periodically (to save CPU time)
            if (epoch % args.eval_every) == 0:
                va_loss, va_scores = evaluate(
                    model, val_loader, device, args.num_classes, log_interval=0
                )
            else:
                va_loss, va_scores = float("nan"), {"OA": float("nan"), "mAcc": float("nan"), "mIoU": float("nan")}

            # Log epoch summary (JSON is nice for later parsing)
            dt = time.time() - t0
            log_epoch = {
                "epoch": epoch,
                "time_sec": round(dt, 2),
                "train_loss": float(tr_loss),
                "train_OA": float(tr_scores["OA"]),
                "train_mAcc": float(tr_scores["mAcc"]),
                "train_mIoU": float(tr_scores["mIoU"]),
                "val_loss": None if np.isnan(va_loss) else float(va_loss),
                "val_OA": None if np.isnan(va_scores["OA"]) else float(va_scores["OA"]),
                "val_mAcc": None if np.isnan(va_scores["mAcc"]) else float(va_scores["mAcc"]),
                "val_mIoU": None if np.isnan(va_scores["mIoU"]) else float(va_scores["mIoU"]),
            }
            print(json.dumps(log_epoch), flush=True)
            history.append(log_epoch)

            # Checkpoints: always save "last", update "best" on val mIoU
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "last_model.pth"))
            if log_epoch["val_mIoU"] is not None:
                if log_epoch["val_mIoU"] > best_miou:
                    best_miou = log_epoch["val_mIoU"]
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pth"))
        
        # Step the scheduler (if not done per optimizer step)
        scheduler.step()

        # Save training history
        with open(os.path.join(args.outdir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    except KeyboardInterrupt:
        # Graceful stop with checkpoint
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "interrupt_model.pth"))
        print(json.dumps({"event": "interrupt", "msg": "Saved interrupt_model.pth"}), flush=True)


if __name__ == "__main__":
    # Windows safety (esp. if you later experiment with num_workers>0)
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass
    main()
