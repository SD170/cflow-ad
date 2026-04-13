#!/usr/bin/env python3
"""
Run CFLOW-AD inference on the test split; export per-image CSV + heatmap overlays.

Supports --dataset mvtec | btad (same layouts as training).

Run from repo root, e.g.:
  python scripts/infer_export.py --dataset btad --class-name 01 -inp 512 \\
    --checkpoint weights/btad_wide_resnet50_2_freia-cflow_pl3_cb8_inp512_run0_01_<timestamp>.pt \\
    --csv-out inference_outputs/01_scores.csv --heatmap-dir inference_outputs/heatmaps/01
"""
from __future__ import print_function

import argparse
import csv
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve

# Repo root = parent of scripts/
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import timm
from timm.data import resolve_data_config

from custom_datasets import MVTecDataset
from custom_models.utils import load_weights
from model import load_decoder_arch, load_encoder_arch
from train import test_meta_epoch


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def prepare_config(args):
    c = argparse.Namespace(**vars(args))
    c.model = "{}_{}_{}_pl{}_cb{}_inp{}_run{}_{}".format(
        c.dataset,
        c.enc_arch,
        c.dec_arch,
        c.pool_layers,
        c.coupling_blocks,
        c.input_size,
        c.run_name,
        c.class_name,
    )
    if ("vit" in c.enc_arch) or ("efficient" in c.enc_arch):
        encoder = timm.create_model(c.enc_arch, pretrained=True)
        arch_config = resolve_data_config({}, model=encoder)
        c.norm_mean, c.norm_std = list(arch_config["mean"]), list(arch_config["std"])
        c.img_size = arch_config["input_size"][1:]
        c.crp_size = arch_config["input_size"][1:]
    else:
        c.img_size = (c.input_size, c.input_size)
        c.crp_size = (c.input_size, c.input_size)
        if c.dataset == "stc":
            c.norm_mean, c.norm_std = 3 * [0.5], 3 * [0.225]
        else:
            c.norm_mean, c.norm_std = (
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            )
    c.img_dims = [3] + list(c.img_size)
    c.clamp_alpha = 1.9
    c.condition_vec = 128
    c.dropout = 0.0
    c.print_freq = 2
    c.temp = 0.5
    c.lr_decay_epochs = [i * c.meta_epochs // 100 for i in [50, 75, 90]]
    c.lr_decay_rate = 0.1
    c.lr_warm_epochs = 2
    c.lr_warm = True
    c.lr_cosine = True
    c.lr_warmup_from = c.lr / 10.0
    if c.lr_cosine:
        eta_min = c.lr * (c.lr_decay_rate ** 3)
        c.lr_warmup_to = eta_min + (c.lr - eta_min) * (
            1 + math.cos(math.pi * c.lr_warm_epochs / c.meta_epochs)
        ) / 2
    else:
        c.lr_warmup_to = c.lr

    if c.dataset == "mvtec":
        c.data_path = "./data/MVTec-AD"
    elif c.dataset == "btad":
        c.data_path = "./data/BTAD/BTech_Dataset_transformed"
    elif c.dataset == "stc":
        c.data_path = "./data/STC/shanghaitech"
    else:
        raise NotImplementedError("dataset {} not supported".format(c.dataset))

    os.environ["CUDA_VISIBLE_DEVICES"] = c.gpu
    c.use_cuda = not c.no_cuda and torch.cuda.is_available()
    init_seeds(0)
    c.device = torch.device("cuda" if c.use_cuda else "cpu")
    c.verbose = True
    c.hide_tqdm_bar = True
    c.save_results = False
    c.viz = False
    return c


def compute_super_mask(c, test_dist, height, width, pool_layers):
    test_map = [list() for _ in pool_layers]
    for l, _ in enumerate(pool_layers):
        test_norm = torch.tensor(test_dist[l], dtype=torch.double)
        test_norm -= torch.max(test_norm)
        test_prob = torch.exp(test_norm)
        test_mask = test_prob.reshape(-1, height[l], width[l])
        test_map[l] = (
            F.interpolate(
                test_mask.unsqueeze(1),
                size=c.crp_size,
                mode="bilinear",
                align_corners=True,
            )
            .squeeze()
            .numpy()
        )
    score_map = np.zeros_like(test_map[0])
    for l, _ in enumerate(pool_layers):
        score_map += test_map[l]
    super_mask = score_map.max() - score_map
    score_label = np.max(super_mask, axis=(1, 2))
    return super_mask, score_label


def compute_image_threshold(gt_labels, score_label):
    precision, recall, thresholds = precision_recall_curve(
        gt_labels.astype(bool), score_label
    )
    if len(thresholds) == 0:
        return float(np.median(score_label))
    a = 2 * precision[:-1] * recall[:-1]
    b = precision[:-1] + recall[:-1]
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx])


def to_uint8_heatmap(arr):
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    denom = arr.max()
    if denom > 0:
        arr = arr / denom
    return (arr * 255.0).clip(0, 255).astype(np.uint8)


def save_overlay(image_path, heatmap, out_path, alpha=0.45):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    img = img.resize((heatmap.shape[1], heatmap.shape[0]), Image.LANCZOS)
    hm = Image.fromarray(to_uint8_heatmap(heatmap), mode="L")
    red_overlay = np.zeros((heatmap.shape[0], heatmap.shape[1], 3), dtype=np.uint8)
    red_overlay[..., 0] = np.array(hm, dtype=np.uint8)
    img_np = np.array(img, dtype=np.float32)
    overlay_np = (1 - alpha) * img_np + alpha * red_overlay.astype(np.float32)
    overlay_np = overlay_np.clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay_np).save(out_path)


def path_context(dataset, class_name, image_path):
    """Return (defect_type_from_path, category_folder, subfolder) for CSV."""
    p = Path(image_path)
    if dataset == "btad":
        parts = p.parts
        try:
            ti = parts.index("test")
            tail = list(parts[ti + 1 :])
        except ValueError:
            tail = [p.parent.parent.name, p.parent.name, p.name]
        if len(tail) >= 2:
            category = tail[0]
            subfolder = tail[1] if len(tail) > 2 else ""
        else:
            category, subfolder = "", ""
        defect_type = "{}_{}".format(category, subfolder) if subfolder else category
        return defect_type, category, subfolder
    category = p.parent.name
    return category, category, ""


def main():
    parser = argparse.ArgumentParser(description="CFLOW-AD inference CSV + heatmaps")
    parser.add_argument("--dataset", default="btad", choices=["mvtec", "btad"])
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("-cl", "--class-name", required=True, type=str)
    parser.add_argument("-enc", "--enc-arch", default="wide_resnet50_2", type=str)
    parser.add_argument("-dec", "--dec-arch", default="freia-cflow", type=str)
    parser.add_argument("-pl", "--pool-layers", default=3, type=int)
    parser.add_argument("-cb", "--coupling-blocks", default=8, type=int)
    parser.add_argument("-run", "--run-name", default=0, type=int)
    parser.add_argument("-inp", "--input-size", default=512, type=int)
    parser.add_argument("-bs", "--batch-size", default=32, type=int)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--meta-epochs", type=int, default=25)
    parser.add_argument("--sub-epochs", type=int, default=8)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument(
        "--csv-out", default="inference_outputs/scores.csv", type=str
    )
    parser.add_argument("--heatmap-dir", default="inference_outputs/heatmaps", type=str)
    parser.add_argument(
        "--threshold",
        default=None,
        type=float,
        help="If set, use this image-level score threshold instead of F1 on test set",
    )
    args = parser.parse_args()

    os.chdir(_REPO_ROOT)
    c = prepare_config(args)

    encoder, pool_layers, pool_dims = load_encoder_arch(c, c.pool_layers)
    encoder = encoder.to(c.device).eval()
    decoders = [load_decoder_arch(c, pd).to(c.device) for pd in pool_dims]

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        ckpt = _REPO_ROOT / args.checkpoint
    if not ckpt.is_file():
        raise FileNotFoundError("checkpoint not found: {}".format(args.checkpoint))
    load_weights(encoder, decoders, str(ckpt.resolve()))

    kwargs = {"num_workers": c.workers, "pin_memory": True} if c.use_cuda else {}
    test_dataset = MVTecDataset(c, is_train=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=c.batch_size,
        shuffle=False,
        drop_last=False,
        **kwargs
    )

    N = 256
    height, width, _, test_dist, gt_label_list, gt_mask_list = test_meta_epoch(
        c, 0, test_loader, encoder, decoders, pool_layers, N
    )

    super_mask, score_label = compute_super_mask(
        c, test_dist, height, width, pool_layers
    )
    gt_label = np.asarray(gt_label_list, dtype=bool)
    if args.threshold is not None:
        image_threshold = float(args.threshold)
    else:
        image_threshold = compute_image_threshold(gt_label, score_label)

    csv_out = Path(args.csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    heatmap_dir = Path(args.heatmap_dir)
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(test_dataset.x)
    labels = list(test_dataset.y)
    mask_paths = list(test_dataset.mask)

    fieldnames = [
        "product_class",
        "image_path",
        "defect_type_from_path",
        "category_folder",
        "subfolder",
        "gt_is_anomalous",
        "gt_mask_path",
        "anomaly_score",
        "image_threshold_used",
        "predicted_anomalous",
        "prediction_correct",
        "heatmap_path",
    ]

    with csv_out.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for i, img_path in enumerate(image_paths):
            img_path_str = str(img_path)
            defect_type, category, subfolder = path_context(
                c.dataset, c.class_name, img_path_str
            )
            gt_is_anomalous = bool(labels[i]) if i < len(labels) else False
            if i < len(mask_paths) and mask_paths[i] not in (None, 0, "0"):
                gt_mask_path = str(mask_paths[i])
            else:
                gt_mask_path = ""

            score = float(score_label[i])
            predicted = bool(score > image_threshold)
            correct = predicted == gt_is_anomalous

            safe_stem = "{}_{}_{}".format(category, subfolder, Path(img_path_str).stem)
            safe_stem = safe_stem.replace(os.sep, "_").replace(" ", "_")
            heatmap_path = heatmap_dir / "{}_heatmap.png".format(safe_stem)
            save_overlay(img_path_str, super_mask[i], heatmap_path)

            writer.writerow(
                {
                    "product_class": c.class_name,
                    "image_path": img_path_str,
                    "defect_type_from_path": defect_type,
                    "category_folder": category,
                    "subfolder": subfolder,
                    "gt_is_anomalous": int(gt_is_anomalous),
                    "gt_mask_path": gt_mask_path,
                    "anomaly_score": score,
                    "image_threshold_used": image_threshold,
                    "predicted_anomalous": int(predicted),
                    "prediction_correct": int(correct),
                    "heatmap_path": str(heatmap_path.resolve()),
                }
            )

    print("Saved CSV: {}".format(csv_out.resolve()))
    print("Heatmaps: {}".format(heatmap_dir.resolve()))
    print("Threshold: {:.6f} ({})".format(
        image_threshold,
        "fixed" if args.threshold is not None else "F1-optimal on test set",
    ))


if __name__ == "__main__":
    main()
