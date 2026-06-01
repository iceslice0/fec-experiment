#!/usr/bin/env python3
"""
CLIP semantic evaluation for an original/reconstruction montage.

Expected montage layout:
    - 8 columns
    - square tiles, default 226 x 226
    - outer padding only, default 1 px
    - no separators between tiles
    - 0-based columns 0,2,4,6 are originals
    - 0-based columns 1,3,5,7 are reconstructions
    - each reconstruction corresponds to the original immediately to its left

Example:
    python clip_montage_eval.py best_orig_vs_recon_all.png

Optional:
    python clip_montage_eval.py best_orig_vs_recon_all.png --save-crops crops
"""

import argparse
import csv
import json
import math
from pathlib import Path

import torch
from PIL import Image

try:
    import open_clip
except ImportError as exc:
    raise ImportError(
        "Missing dependency: open_clip_torch. Install with:\n"
        "    pip install open_clip_torch"
    ) from exc


def split_orig_recon_montage_fixed(
    montage_path,
    tile_size=226,
    pad=1,
    ncols=8,
    save_dir=None,
):
    """
    Split fixed-grid montage into paired originals/reconstructions.
    No internal separator is assumed.
    """

    montage_path = Path(montage_path)
    img = Image.open(montage_path).convert("RGB")
    W, H = img.size

    expected_W = ncols * tile_size + 2 * pad

    if W != expected_W:
        raise ValueError(
            f"Unexpected montage width: got W={W}, expected {expected_W}. "
            f"Expected {ncols} columns of {tile_size}px plus {pad}px padding "
            f"on both sides."
        )

    if (H - 2 * pad) % tile_size != 0:
        raise ValueError(
            f"Unexpected montage height: got H={H}. "
            f"H - 2*pad = {H - 2 * pad} must be divisible by tile_size={tile_size}."
        )

    nrows = (H - 2 * pad) // tile_size

    if save_dir is not None:
        save_dir = Path(save_dir)
        orig_dir = save_dir / "originals"
        rec_dir = save_dir / "reconstructions"
        orig_dir.mkdir(parents=True, exist_ok=True)
        rec_dir.mkdir(parents=True, exist_ok=True)
    else:
        orig_dir = None
        rec_dir = None

    originals = []
    reconstructions = []
    pair_meta = []

    pair_id = 0

    for r in range(nrows):
        y0 = pad + r * tile_size
        y1 = y0 + tile_size

        for c_orig in range(0, ncols, 2):
            c_rec = c_orig + 1

            x0_orig = pad + c_orig * tile_size
            x1_orig = x0_orig + tile_size

            x0_rec = pad + c_rec * tile_size
            x1_rec = x0_rec + tile_size

            orig = img.crop((x0_orig, y0, x1_orig, y1))
            rec = img.crop((x0_rec, y0, x1_rec, y1))

            if save_dir is None:
                originals.append(orig)
                reconstructions.append(rec)
                orig_path = ""
                rec_path = ""
            else:
                orig_path = orig_dir / f"orig_{pair_id:04d}.png"
                rec_path = rec_dir / f"rec_{pair_id:04d}.png"

                orig.save(orig_path)
                rec.save(rec_path)

                originals.append(str(orig_path))
                reconstructions.append(str(rec_path))

            pair_meta.append(
                {
                    "pair_id": pair_id,
                    "row": r,
                    "orig_col": c_orig,
                    "rec_col": c_rec,
                    "orig_box": [x0_orig, y0, x1_orig, y1],
                    "rec_box": [x0_rec, y0, x1_rec, y1],
                    "orig_path": str(orig_path),
                    "rec_path": str(rec_path),
                }
            )

            pair_id += 1

    grid_info = {
        "montage_path": str(montage_path),
        "montage_width": W,
        "montage_height": H,
        "nrows": nrows,
        "ncols": ncols,
        "tile_size": tile_size,
        "pad": pad,
        "num_pairs": len(originals),
    }

    return originals, reconstructions, pair_meta, grid_info


@torch.no_grad()
def encode_images(model, preprocess, images_or_paths, device, batch_size=64):
    all_feats = []

    for start in range(0, len(images_or_paths), batch_size):
        batch_items = images_or_paths[start:start + batch_size]
        imgs = []

        for item in batch_items:
            if isinstance(item, Image.Image):
                img = item.convert("RGB")
            else:
                img = Image.open(item).convert("RGB")

            imgs.append(preprocess(img))

        imgs = torch.stack(imgs).to(device)

        feats = model.encode_image(imgs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu())

    return torch.cat(all_feats, dim=0)


def safe_std(x):
    if x.numel() <= 1:
        return 0.0
    return x.std(unbiased=True).item()


def pvalue_matched_gt_random(matched_scores, row_mean_nonmatch):
    """
    One-sided p-value for matched CLIP scores exceeding non-matching scores
    for the same original (H1: matched > mean non-match per row).
    """
    diffs = (matched_scores - row_mean_nonmatch).cpu().numpy()

    try:
        from scipy.stats import wilcoxon

        if (diffs == 0).all():
            return 1.0

        return float(
            wilcoxon(diffs, alternative="greater", zero_method="wilcox").pvalue
        )
    except ImportError:
        diffs_nz = diffs[diffs != 0]
        n = len(diffs_nz)
        if n == 0:
            return 1.0
        k = int((diffs_nz > 0).sum())
        return sum(math.comb(n, i) for i in range(k, n + 1)) / (2**n)


def format_pvalue(p):
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def compute_image_image_metrics(orig_feats, rec_feats):
    S = orig_feats @ rec_feats.T
    N = S.shape[0]
    if N <= 1:
        raise ValueError(f"Need at least 2 pairs for similarity metrics, got N={N}")

    off_diag = ~torch.eye(N, dtype=torch.bool, device=S.device)
    row_mean_nonmatch = (S.sum(dim=1) - S.diag()) / (N - 1)
    col_mean_nonmatch = (S.sum(dim=0) - S.diag()) / (N - 1)
    global_mean_nonmatch = S[off_diag].mean()

    S_norm = S / (row_mean_nonmatch.unsqueeze(1) + 1e-8)
    S_norm = S_norm / (col_mean_nonmatch.unsqueeze(0) + 1e-8)
    S_norm = S_norm * (global_mean_nonmatch ** 2)

    correct = torch.arange(N).unsqueeze(1)
    k5 = min(5, N)

    def eval_matrix(M, row_mean_for_p):
        matched_scores = M.diag()
        random_scores = M[~torch.eye(N, dtype=torch.bool, device=M.device)]
        ranks = torch.argsort(M, dim=1, descending=True)
        positions = (ranks == correct).nonzero()[:, 1] + 1
        positions = positions.float()
        mean_random = random_scores.mean().item()
        std_random = safe_std(random_scores)
        mean_matched = matched_scores.mean().item()
        std_matched = safe_std(matched_scores)

        return {
            "clip_image_image_matched_mean": mean_matched,
            "clip_image_image_matched_std": std_matched,
            "clip_random_mismatch_mean": mean_random,
            "clip_random_mismatch_std": std_random,
            "clip_semantic_gap": mean_matched - mean_random,
            "clip_pvalue_matched_gt_random": pvalue_matched_gt_random(
                matched_scores, row_mean_for_p
            ),
            "retrieval_recall_at_1": (ranks[:, :1] == correct).any(dim=1).float().mean().item(),
            "retrieval_recall_at_5": (ranks[:, :k5] == correct).any(dim=1).float().mean().item(),
            "median_rank": positions.median().item(),
            "mean_rank": positions.mean().item(),
            "mrr": (1.0 / positions).mean().item(),
            "_matched_scores": matched_scores,
            "_positions": positions,
        }

    metrics_s = eval_matrix(S, row_mean_nonmatch)
    row_mean_norm = (S_norm.sum(dim=1) - S_norm.diag()) / (N - 1)
    metrics_norm = eval_matrix(S_norm, row_mean_norm)

    metrics = {
        "num_pairs": N,
        "global_mean_nonmatch": global_mean_nonmatch.item(),
        "S": {k: v for k, v in metrics_s.items() if not k.startswith("_")},
        "S_norm": {k: v for k, v in metrics_norm.items() if not k.startswith("_")},
    }

    per_pair = []
    for i in range(N):
        per_pair.append(
            {
                "pair_id": i,
                "matched_clip_similarity": metrics_s["_matched_scores"][i].item(),
                "retrieval_rank": int(metrics_s["_positions"][i].item()),
                "matched_clip_similarity_norm": metrics_norm["_matched_scores"][i].item(),
                "retrieval_rank_norm": int(metrics_norm["_positions"][i].item()),
            }
        )

    return metrics, per_pair, {"S": S, "S_norm": S_norm}


def write_per_pair_csv(path, pair_meta, per_pair):
    path = Path(path)

    rows = []
    for meta, pp in zip(pair_meta, per_pair):
        row = dict(meta)
        row.update(pp)
        rows.append(row)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP semantic similarity for original/reconstruction montage."
    )

    parser.add_argument(
        "montage",
        type=str,
        help="Path to montage PNG.",
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=226,
        help="Tile size in pixels. Default: 226.",
    )

    parser.add_argument(
        "--pad",
        type=int,
        default=1,
        help="Outer montage padding in pixels. Default: 1.",
    )

    parser.add_argument(
        "--ncols",
        type=int,
        default=8,
        help="Number of montage columns. Default: 8.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-32",
        help="OpenCLIP model name. Default: ViT-B-32.",
    )

    parser.add_argument(
        "--pretrained",
        type=str,
        default="openai",
        help="OpenCLIP pretrained weights. Default: openai.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for CLIP encoding. Default: 64.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda, cpu, etc. Default: cuda if available else cpu.",
    )

    parser.add_argument(
        "--save-crops",
        type=str,
        default=None,
        help="Optional directory to save extracted original/reconstruction crops.",
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default="clip_montage_metrics.json",
        help="Output JSON file. Default: clip_montage_metrics.json.",
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        default="clip_montage_per_pair.csv",
        help="Output per-pair CSV file. Default: clip_montage_per_pair.csv.",
    )

    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    originals, reconstructions, pair_meta, grid_info = split_orig_recon_montage_fixed(
        montage_path=args.montage,
        tile_size=args.tile_size,
        pad=args.pad,
        ncols=args.ncols,
        save_dir=args.save_crops,
    )

    print("Montage geometry")
    print("----------------")
    print(f"File:       {grid_info['montage_path']}")
    print(f"Size:       {grid_info['montage_width']} x {grid_info['montage_height']}")
    print(f"Grid:       {grid_info['nrows']} rows x {grid_info['ncols']} columns")
    print(f"Tile size:  {grid_info['tile_size']} x {grid_info['tile_size']}")
    print(f"Outer pad:  {grid_info['pad']} px")
    print(f"Pairs:      {grid_info['num_pairs']}")
    print()

    print("Loading CLIP")
    print("------------")
    print(f"Model:      {args.model}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Device:     {device}")
    print()

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model,
        pretrained=args.pretrained,
    )
    model = model.to(device)
    model.eval()

    print("Encoding images...")
    orig_feats = encode_images(
        model=model,
        preprocess=preprocess,
        images_or_paths=originals,
        device=device,
        batch_size=args.batch_size,
    )

    rec_feats = encode_images(
        model=model,
        preprocess=preprocess,
        images_or_paths=reconstructions,
        device=device,
        batch_size=args.batch_size,
    )

    metrics, per_pair, _ = compute_image_image_metrics(orig_feats, rec_feats)

    result = {
        "grid_info": grid_info,
        "clip_model": args.model,
        "clip_pretrained": args.pretrained,
        "device": device,
        "metrics": metrics,
    }

    with Path(args.output_json).open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    write_per_pair_csv(args.output_csv, pair_meta, per_pair)

    print()
    print("CLIP semantic metrics")
    print("=====================")
    print(f"N pairs:                         {metrics['num_pairs']}")
    print(
        "Global mean non-match (raw S):   "
        f"{metrics['global_mean_nonmatch']:.4f}"
    )
    for title, block in (
        ("Raw similarity (S)", metrics["S"]),
        ("Row/col normalized (S_norm)", metrics["S_norm"]),
    ):
        print()
        print(title)
        print("-" * len(title))
        print(
            "CLIP image-image matched:        "
            f"{block['clip_image_image_matched_mean']:.4f} "
            f"± {block['clip_image_image_matched_std']:.4f}"
        )
        print(
            "CLIP random mismatch:            "
            f"{block['clip_random_mismatch_mean']:.4f} "
            f"± {block['clip_random_mismatch_std']:.4f}"
        )
        print(
            "CLIP semantic gap:               "
            f"{block['clip_semantic_gap']:.4f}"
        )
        print(
            "p-value (matched > random):      "
            f"{format_pvalue(block['clip_pvalue_matched_gt_random'])}"
        )
        print(
            "Retrieval Recall@1:              "
            f"{100.0 * block['retrieval_recall_at_1']:.2f}%"
        )
        print(
            "Retrieval Recall@5:              "
            f"{100.0 * block['retrieval_recall_at_5']:.2f}%"
        )
        print(f"Median rank:                     {block['median_rank']:.1f}")
        print(f"Mean rank:                       {block['mean_rank']:.2f}")
        print(f"MRR:                             {block['mrr']:.4f}")
    print()
    print(f"Wrote JSON: {args.output_json}")
    print(f"Wrote CSV:  {args.output_csv}")


if __name__ == "__main__":
    main()