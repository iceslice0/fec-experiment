#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
invert_regnet_imagenet.py

Invert ImageNet-1K samples from RegNet trunk-output features via gradient-based optimization.
Uses only conv/ReLU/skip/pool/linear layers. Saves original and reconstructed batches.
"""
import argparse
import os
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, utils, models
from torchvision.datasets import ImageNet

from math import sqrt, pi, erf, inf


REGNET_MODELS = {
    "regnet_x_3_2gf": (
        models.regnet_x_3_2gf,
        models.RegNet_X_3_2GF_Weights.DEFAULT,
    ),
    "regnet_y_3_2gf": (
        models.regnet_y_3_2gf,
        models.RegNet_Y_3_2GF_Weights.DEFAULT,
    ),
}


def load_regnet_model(name: str):
    factory, weights = REGNET_MODELS[name]
    return factory(weights=weights)


def replace_relu_with_softplus(module: nn.Module, beta: float = 1.0, threshold: float = 20.0):
    """Recursively replace ReLU with Softplus"""
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.Softplus(beta=beta, threshold=threshold))
        else:
            replace_relu_with_softplus(child, beta=beta, threshold=threshold)


def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB tensor to HSV.
    
    Args:
        rgb: (B, 3, H, W) tensor in [0, 1] range
        
    Returns:
        hsv: (B, 3, H, W) tensor with H in [0, 1], S in [0, 1], V in [0, 1]
    """
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    
    max_val, max_idx = torch.max(rgb, dim=1)
    min_val = torch.min(rgb, dim=1)[0]
    delta = max_val - min_val
    
    # Value (brightness)
    v = max_val
    
    # Saturation
    s = torch.where(max_val > 1e-6, delta / max_val, torch.zeros_like(max_val))
    
    # Hue
    h = torch.zeros_like(max_val)
    mask_r = (max_idx == 0) & (delta > 1e-6)
    mask_g = (max_idx == 1) & (delta > 1e-6)
    mask_b = (max_idx == 2) & (delta > 1e-6)
    
    h[mask_r] = (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6) / 6.0
    h[mask_g] = ((b[mask_g] - r[mask_g]) / delta[mask_g] + 2) / 6.0
    h[mask_b] = ((r[mask_b] - g[mask_b]) / delta[mask_b] + 4) / 6.0
    
    hsv = torch.stack([h, s, v], dim=1)
    return hsv


def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    """
    Convert HSV tensor to RGB.
    
    Args:
        hsv: (B, 3, H, W) tensor with H in [0, 1], S in [0, 1], V in [0, 1]
        
    Returns:
        rgb: (B, 3, H, W) tensor in [0, 1] range
    """
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    
    c = v * s
    x = c * (1 - torch.abs((h * 6) % 2 - 1))
    m = v - c
    
    h6 = (h * 6) % 6
    sector = h6.floor().long()
    
    # Initialize RGB channels
    r = torch.zeros_like(c)
    g = torch.zeros_like(c)
    b = torch.zeros_like(c)
    
    # Map sectors to RGB values
    mask0 = (sector == 0)
    mask1 = (sector == 1)
    mask2 = (sector == 2)
    mask3 = (sector == 3)
    mask4 = (sector == 4)
    mask5 = (sector == 5)
    
    r[mask0] = c[mask0]
    g[mask0] = x[mask0]
    b[mask0] = 0
    
    r[mask1] = x[mask1]
    g[mask1] = c[mask1]
    b[mask1] = 0
    
    r[mask2] = 0
    g[mask2] = c[mask2]
    b[mask2] = x[mask2]
    
    r[mask3] = 0
    g[mask3] = x[mask3]
    b[mask3] = c[mask3]
    
    r[mask4] = x[mask4]
    g[mask4] = 0
    b[mask4] = c[mask4]
    
    r[mask5] = c[mask5]
    g[mask5] = 0
    b[mask5] = x[mask5]
    
    rgb = torch.stack([r + m, g + m, b + m], dim=1)
    return rgb.clamp_(0., 1.)


def grayworld_white_balance(batch: torch.Tensor) -> torch.Tensor:
    """
    Apply gray world white balance to each image in batch.
    The gray world assumption states that the average color of an image should be gray.
    This neutralizes color casts by scaling RGB channels so their means are equal.
    
    Args:
        batch: (B, 3, H, W) tensor in [0, 1] range
        
    Returns:
        White-balanced batch with neutralized color casts
    """
    # Compute mean of each RGB channel for each image: (B, 3)
    r_mean = batch[:, 0, :, :].mean(dim=[1, 2])  # (B,)
    g_mean = batch[:, 1, :, :].mean(dim=[1, 2])  # (B,)
    b_mean = batch[:, 2, :, :].mean(dim=[1, 2])  # (B,)
    
    # Compute overall mean (average of R, G, B means) for each image: (B,)
    overall_mean = (r_mean + g_mean + b_mean) / 3.0
    
    # Compute scaling factors to make each channel mean equal to overall mean
    # Avoid division by zero
    r_scale = (overall_mean / r_mean.clamp(min=1e-6)).view(batch.size(0), 1, 1)
    g_scale = (overall_mean / g_mean.clamp(min=1e-6)).view(batch.size(0), 1, 1)
    b_scale = (overall_mean / b_mean.clamp(min=1e-6)).view(batch.size(0), 1, 1)
    
    # Apply scaling to each channel (broadcasting works with (B, 1, 1) * (B, H, W))
    balanced = torch.stack([
        batch[:, 0, :, :] * r_scale,
        batch[:, 1, :, :] * g_scale,
        batch[:, 2, :, :] * b_scale
    ], dim=1)
    
    return balanced.clamp_(0., 1.)


def normalize_contrast_saturation(batch: torch.Tensor) -> torch.Tensor:
    """
    Normalize each image in batch by saturation (S) and value (V) in HSV space.
    Converts RGB to HSV, normalizes S and V using 0.01 and 0.99 quantiles (keeping original hue),
    then converts back to RGB.
    
    Args:
        batch: (B, 3, H, W) tensor in [0, 1] range
        
    Returns:
        Normalized batch with maximized contrast and saturation per image
    """
    # Convert RGB to HSV
    hsv = rgb_to_hsv(batch)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    
    # Helper function to normalize a channel using quantiles
    def normalize_channel(channel, l, h):
        channel_flat = channel.view(batch.size(0), -1)
        q01 = torch.quantile(channel_flat, l, dim=1, keepdim=True).view(batch.size(0), 1, 1)
        q99 = torch.quantile(channel_flat, h, dim=1, keepdim=True).view(batch.size(0), 1, 1)
        channel_range = (q99 - q01).clamp(min=5e-2)
        return ((channel - q01) / channel_range).clamp_(0., 1.)
    
    # Normalize S (saturation) and V (value/brightness) for each image
    # Keep original H (hue) - hue shifting causes color cast issues
    s_normalized = normalize_channel(s, 0.0, 1.0)
    v_normalized = normalize_channel(v, 0.05, 0.95).clamp(min=1e-6)

    # Apply gamma 
    mean_v_n = v_normalized.mean(dim=[1, 2], keepdim=True)
    mean_v = v.mean(dim=[1, 2], keepdim=True).clamp(min=1e-6)
    gamma = torch.log(mean_v) / torch.log(mean_v_n)
    gamma = gamma.clamp(0.1, 5.0)  # avoid extremes
    v_adjusted = v_normalized.pow(gamma).clamp_(0., 1.)
    
    # Reconstruct HSV with original H, normalized S and V
    hsv_normalized = torch.stack([h, s_normalized, v_adjusted], dim=1)
    
    # Convert back to RGB
    normalized = hsv_to_rgb(hsv_normalized)
    
    return normalized.clamp_(0., 1.)


def reduce_batch_lp(losses: torch.Tensor, p: float = 2.0) -> torch.Tensor:
    """L_p reduction over batch: (mean_i |loss_i|^p)^(1/p), batch-size independent."""
    losses = losses.reshape(-1).abs().pow(p)
    return losses.mean().pow(1.0 / p)


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    """Total variation loss per sample, shape (B,)."""
    dh = x[:, :, 1:, :] - x[:, :, :-1, :]
    dw = x[:, :, :, 1:] - x[:, :, :, :-1]
    dhr = dh[:, 0, :, :] - dh[:, 1, :, :]
    dhb = dh[:, 2, :, :] - dh[:, 1, :, :]
    dwr = dw[:, 0, :, :] - dw[:, 1, :, :]
    dwb = dw[:, 2, :, :] - dw[:, 1, :, :]
    return (
        dh.abs().sum(dim=[1, 2, 3]) + dw.abs().sum(dim=[1, 2, 3])
        + dhr.abs().sum(dim=[1, 2]) + dhb.abs().sum(dim=[1, 2])
        + dwr.abs().sum(dim=[1, 2]) + dwb.abs().sum(dim=[1, 2])
    )


def sym_kl_div(x, y) -> torch.Tensor:
    """Symmetric KL divergence per sample, shape (B,)."""
    x = F.log_softmax(x, dim=1)
    y = F.log_softmax(y, dim=1)
    kl1 = F.kl_div(x, y, log_target=True, reduction='none').sum(dim=[1, 2, 3])
    kl2 = F.kl_div(y, x, log_target=True, reduction='none').sum(dim=[1, 2, 3])
    return 0.5 * (kl1 + kl2)


def gray_edge_per_sample(rgb: torch.Tensor) -> torch.Tensor:
    """Gray-edge regularizer per sample, shape (B,). rgb: (B,3,H,W) ImageNet-normalized."""
    dx = rgb[:, :, :, 1:] - rgb[:, :, :, :-1]
    dy = rgb[:, :, 1:, :] - rgb[:, :, :-1, :]

    def dev_from_mean(d):
        m = d.mean(dim=1, keepdim=True)
        return (d - m).abs().sum(dim=1)

    return dev_from_mean(dx).mean(dim=[1, 2]) + dev_from_mean(dy).mean(dim=[1, 2])


def input_norm_moments_per_sample(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Teacher-free moment priors on the student input x (B, 3, H, W), ImageNet-normalized.
    Penalize per-channel spatial mean away from 0 and std away from 1.
    """
    mean_ps = x.mean(dim=[2, 3]).pow(2).mean(dim=1)
    std_ps = (x.std(dim=[2, 3], unbiased=False) - 1.0).pow(2).mean(dim=1)
    return mean_ps, std_ps


def project_input_spatial(x: torch.Tensor, eps: float = 1e-5) -> None:
    """In-place: per (batch, channel), zero spatial mean and unit spatial std."""
    mu = x.mean(dim=(2, 3), keepdim=True)
    sig = x.std(dim=(2, 3), keepdim=True, unbiased=False).clamp(min=eps)
    x.sub_(mu).div_(sig)


def compute_all_losses(
    feat,
    target_feat,
    x,
    tv_weight,
    l2_weight=0.0,
    mean_weight=0.0,
    std_weight=0.0,
    l1_weight=0.0,
    tv_loss_orig=None,
    batch_norm_p=2.0,
):
    """Compute per-sample losses, then L_p reduce over batch (batch-size independent)."""
    kl_ps = sym_kl_div(feat, target_feat)
    feat_mean = feat.mean(dim=[1, 2, 3])
    target_mean = target_feat.mean(dim=[1, 2, 3])
    mse_ps = (feat_mean - target_mean).pow(2)
    mean_ps, std_ps = input_norm_moments_per_sample(x)

    tv_ps = tv_loss(x)
    if tv_loss_orig is not None:
        tv_ps = torch.maximum(tv_ps - 0.4 * tv_loss_orig, torch.zeros_like(tv_ps))
    edge_ps = gray_edge_per_sample(x)

    loss_kl = reduce_batch_lp(kl_ps, batch_norm_p)
    loss_mse = l2_weight * reduce_batch_lp(mse_ps, batch_norm_p)
    loss_mean = mean_weight * reduce_batch_lp(mean_ps, batch_norm_p)
    loss_std = std_weight * reduce_batch_lp(std_ps, batch_norm_p)
    loss_tv = tv_weight * reduce_batch_lp(tv_ps, batch_norm_p)
    loss_l1 = l1_weight * reduce_batch_lp(edge_ps, batch_norm_p)

    centering_loss, border_loss = centering_losses(x)

    total_loss = loss_kl + loss_mse + loss_mean + loss_std + loss_tv + loss_l1
    return (
        loss_kl, loss_mse, loss_mean, loss_std, loss_tv, loss_l1,
        centering_loss, border_loss, total_loss, kl_ps,
    )

def denormalize_and_process(x, mean, std):
    """Denormalize tensor and apply post-processing"""
    x_denorm = (x * std + mean).clamp_(0., 1.)
    #x_denorm = normalize_contrast_saturation(x_denorm)
    return x_denorm

def forward_and_get_feat(model, x, activation):
    """Run model forward pass and extract features"""
    _ = model(x)
    return activation['feat']

def save_reconstructed_images(x, mean, std, output_path, nrow=None):
    """Denormalize, process, and save reconstructed images"""
    x_denorm = denormalize_and_process(x, mean, std)
    if not nrow:
        nrow = int(sqrt(len(x)))
    utils.save_image(x_denorm, output_path, nrow=nrow)


def save_best_images(x_best, imgs_best, mean, std, out_dir, class_id_str, nrow=None):
    """Save interleaved original/recon grid plus separate grids"""
    recon_denorm = denormalize_and_process(x_best, mean, std)
    if not nrow:
        nrow = int(sqrt(len(x_best)))
    # Interleaved: orig, recon, orig, recon, ...
    comparison = torch.stack([imgs_best, recon_denorm], dim=1).reshape(-1, *imgs_best.shape[1:])
    utils.save_image(
        comparison,
        os.path.join(out_dir, f'best_orig_vs_recon_{class_id_str}.png'),
        nrow=nrow,
    )
    # Separate grids
    utils.save_image(
        recon_denorm,
        os.path.join(out_dir, f'best_recon_{class_id_str}.png'),
        nrow=nrow,
    )
    utils.save_image(
        imgs_best,
        os.path.join(out_dir, f'best_orig_{class_id_str}.png'),
        nrow=nrow,
    )


def tv_feature_loss(f):
    # f: (B, C, H, W)
    # whiten
    f = (f - f.mean(dim=[0,2,3], keepdim=True)) / (f.std(dim=[0,2,3], keepdim=True) + 1e-5)
    # compute spatial diffs
    dh = f[:, :, 1:, :-1] - f[:, :, :-1, :-1]  # -> (B, C, H-1, W-1)
    dw = f[:, :, :-1, 1:] - f[:, :, :-1, :-1]  # -> (B, C, H-1, W-1)
    # squared magnitude of the C-dimensional gradient vector
    grad2 = dh.abs().sum() + dw.abs().sum()  # (B, H-1, W-1)
    # isotropic vector-TV
    return grad2


def gradient_edginess(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    x: (B,C,H,W) in float32, normalized as you use in inversion.
    returns s: (B,H,W) >=0
    """
    # simple forward differences (cheap + stable)
    du = x[..., 1:, :] - x[..., :-1, :]
    dv = x[..., :, 1:] - x[..., :, :-1]

    # pad back to (H,W)
    du = F.pad(du, (0, 0, 0, 1))  # pad last row
    dv = F.pad(dv, (0, 1, 0, 0))  # pad last col

    s = torch.sqrt((du * du + dv * dv).sum(dim=1) + eps)  # sum over channels
    return s

def centering_losses(x: torch.Tensor, eps: float = 1e-8):
    """
    returns (L_ctr, L_bord) scalars
    """
    B, C, H, W = x.shape
    s = gradient_edginess(x, eps=eps)  # (B,H,W)

    # coordinates
    ii = torch.arange(H, device=x.device, dtype=x.dtype).view(1, H, 1).expand(B, H, W)
    jj = torch.arange(W, device=x.device, dtype=x.dtype).view(1, 1, W).expand(B, H, W)

    denom = s.sum(dim=(1,2)) + eps
    mu_i = (s * ii).sum(dim=(1,2)) / denom
    mu_j = (s * jj).sum(dim=(1,2)) / denom

    c_i = (H - 1) / 2.0
    c_j = (W - 1) / 2.0

    L_ctr = ((mu_i - c_i) ** 2 + (mu_j - c_j) ** 2).mean()

    # soft border penalty (radius^2)
    di = (ii - c_i) / float(H)
    dj = (jj - c_j) / float(W)
    d2 = di * di + dj * dj
    L_bord = (s * d2).sum(dim=(1,2)).mean() / (s.sum(dim=(1,2)).mean() + eps)

    return L_ctr, L_bord


def filter_and_sort_by_confidence(dataset, indices, model, device, batch_size, mean, std, subset_size=5000):
    """
    Filter correctly classified images and sort by logit confidence difference.
    
    Args:
        dataset: ImageNet dataset
        indices: List of dataset indices to process
        model: Classification model (should be in eval mode)
        device: Device to run inference on
        batch_size: Batch size for processing
        mean: Mean tensor for normalization
        std: Std tensor for normalization
        subset_size: Number of samples to take from indices before filtering (default: 5000)
        
    Returns:
        List of sorted indices (descending by top-1 vs top-2 logit difference)
    """
    # Fix random seed for reproducibility
    random.seed(2147483648)
    print("Random seed:", random.getstate()[1][0])
    
    original_len = len(indices)
    # Shuffle indices and limit to subset_size if specified
    indices_list = list(indices)
    random.shuffle(indices_list)
    if subset_size > 0 and original_len > subset_size:
        indices_list = indices_list[:subset_size]
        print(f"Processing shuffled subset of {subset_size} images from {original_len} total indices")
    
    print("Filtering correctly classified images and computing logit differences...")
    
    subset_temp = Subset(dataset, indices_list)
    loader_temp = DataLoader(subset_temp, batch_size=batch_size, shuffle=False, num_workers=16)
    
    correct_indices = []
    logit_diffs = []
    
    with torch.no_grad():
        for batch_idx, (imgs, targets) in enumerate(loader_temp):
            imgs = imgs.to(device)
            imgs_norm = (imgs - mean) / std
            logits = model(imgs_norm)
            
            # Get top-2 logits
            top2_logits, top2_indices = torch.topk(logits, k=2, dim=1)
            top1_logits = top2_logits[:, 0]
            top2_logits = top2_logits[:, 1]
            diff = top1_logits - top2_logits  # difference between top-1 and top-2 logits
            
            # Check which are correctly classified
            preds = top2_indices[:, 0]
            correct_mask = (preds == targets.to(device))
            
            # Store indices and differences for correctly classified images
            batch_start_idx = batch_idx * batch_size
            for i in range(len(targets)):
                if correct_mask[i]:
                    global_idx = indices_list[batch_start_idx + i]
                    correct_indices.append(global_idx)
                    logit_diffs.append(diff[i].item())
    
    # Sort by logit difference (descending)
    sorted_pairs = sorted(zip(correct_indices, logit_diffs), key=lambda x: x[1], reverse=True)
    sorted_indices = [idx for idx, _ in sorted_pairs]
    
    print(f"Found {len(sorted_indices)} correctly classified images out of {len(indices_list)} total")
    
    return sorted_indices

def balance_classes(indices, targets, class_ids, max_count=None):
    """
    Balance indices across multiple classes by interleaving.
    
    Args:
        indices: List of dataset indices
        targets: List of class labels for each index (e.g., imgset.targets)
        class_ids: List of class IDs to balance across
        max_count: Maximum number of indices to return (None for all)
        
    Returns:
        Balanced list of indices interleaved across classes
    """
    if len(class_ids) <= 1 or (len(class_ids) == 1 and class_ids[0] == -1):
        # No balancing needed for single class or all classes
        result = indices[:max_count] if max_count else indices
        return result
    
    # Group indices by class
    indices_by_class = {cid: [] for cid in class_ids}
    for idx in indices:
        class_label = targets[idx]
        if class_label in indices_by_class:
            indices_by_class[class_label].append(idx)
    
    # Interleave indices to balance classes
    balanced_indices = []
    max_per_class = max(len(indices_by_class[cid]) for cid in class_ids)
    for i in range(max_per_class):
        for cid in sorted(class_ids):
            if i < len(indices_by_class[cid]):
                balanced_indices.append(indices_by_class[cid][i])
    
    # Limit to max_count if specified
    if max_count and len(balanced_indices) > max_count:
        balanced_indices = balanced_indices[:max_count]
    
    return balanced_indices

def select_best_per_class(batch_indices, loss_per_sample, targets, class_ids, select_n):
    """
    Select best samples per class, then balance across classes.
    
    Args:
        batch_indices: List of batch indices (0 to batch_size-1)
        loss_per_sample: Tensor of per-sample losses (shape: [batch_size])
        targets: List of class labels for each batch index (from dataset)
        class_ids: List of class IDs to balance across
        select_n: Total number of samples to select
        
    Returns:
        List of balanced batch indices
    """
    if len(class_ids) <= 1 or (len(class_ids) == 1 and class_ids[0] == -1):
        # No balancing needed, just select top N
        if select_n < len(batch_indices):
            _, best = torch.topk(-loss_per_sample, k=select_n)
            return best.cpu().tolist()
        return list(range(len(batch_indices)))
    
    # Group batch indices by class
    batch_indices_by_class = {cid: [] for cid in class_ids}
    for batch_idx in batch_indices:
        class_label = targets[batch_idx]
        if class_label in batch_indices_by_class:
            batch_indices_by_class[class_label].append(batch_idx)
    
    # Select best samples per class
    samples_per_class = select_n // len(class_ids)
    remainder = select_n % len(class_ids)
    
    selected_indices = []
    for i, cid in enumerate(sorted(class_ids)):
        class_indices = batch_indices_by_class[cid]
        if len(class_indices) == 0:
            continue
        
        # Get losses for this class
        class_losses = loss_per_sample[class_indices]
        # Select best samples from this class
        n_select = samples_per_class + (1 if i < remainder else 0)
        n_select = min(n_select, len(class_indices))
        
        if n_select > 0:
            _, best_class = torch.topk(-class_losses, k=n_select)
            selected_indices.extend([class_indices[idx] for idx in best_class.cpu().tolist()])
    
    # Interleave to balance (in case we have more than needed)
    if len(selected_indices) > select_n:
        # Group by class again and interleave
        selected_by_class = {cid: [] for cid in class_ids}
        for idx in selected_indices:
            class_label = targets[idx]
            if class_label in selected_by_class:
                selected_by_class[class_label].append(idx)
        
        balanced = []
        max_per_class = max(len(selected_by_class[cid]) for cid in class_ids)
        for i in range(max_per_class):
            for cid in sorted(class_ids):
                if i < len(selected_by_class[cid]):
                    balanced.append(selected_by_class[cid][i])
                    if len(balanced) >= select_n:
                        break
            if len(balanced) >= select_n:
                break
        selected_indices = balanced[:select_n]
    
    return selected_indices

def Phi(x: float) -> float:
    # Standard normal CDF
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def relu_normal_mean_std():
    # v = max(0,s), s~N(0,1)
    mu = 1.0 / sqrt(2.0 * pi)
    var = 0.5 - mu * mu
    return mu, sqrt(var)

def ab_for_threshold(t: float):
    """
    Shifted+scaled 2-level binarizer with threshold t:
        v_hat = b + a * 1{v > t},  v = max(0,s), s~N(0,1)
    Choose a,b so that E[v_hat]=E[v] and Std[v_hat]=Std[v].

    Note: t must be >= 0.  For t<0, P(v>t)=1 -> degenerate (can't match nonzero std).
    """
    mu, sigma = relu_normal_mean_std()

    if t < 0:
        raise ValueError("t < 0 => P(v>t)=1 (degenerate). Use t >= 0.")

    p = 0.5 if t == 0 else 1.0 - Phi(t)   # P(v>t) = P(s>t) for t>0

    if not (0.0 < p < 1.0):
        raise ValueError(f"Need 0<p<1, got p={p}. Pick t>0 (or t=0 gives p=0.5).")

    a = sigma / sqrt(p * (1.0 - p))
    b = mu - a * p
    return a, b, p


def main():
    p = argparse.ArgumentParser(description='Invert RegNet features on ImageNet1K')
    p.add_argument('--data-dir', default='../data/imagenet', help='ImageNet root folder')
    p.add_argument(
        '--model',
        default='regnet_x_3_2gf',
        choices=tuple(REGNET_MODELS),
        help='Torchvision RegNet model to invert. Default: regnet_x_3_2gf',
    )
    p.add_argument('--class-id', default='-1', type=str, help='ImageNet class index(es) - single int or comma-separated list (e.g., "0" or "0,1,2" or "-1" for all)')
    p.add_argument('--batch-size', type=int, default=48, help='Batch size')
    p.add_argument('--steps', type=int, default=4000, help='Optimization steps')
    p.add_argument('--lr', type=float, default=0.5, help='Learning rate')
    p.add_argument('--sigma', type=float, default=0.01, help='Init noise scale')
    p.add_argument('--beta', type=float, default=3., help='Softplus beta for inversion model only (teacher/val use ReLU)')
    p.add_argument('--tv-weight', type=float, default=2.5e-3, help='TV regularization weight')
    p.add_argument('--l2-weight', type=float, default=10.0, help='MSE on global feature mean (scalar per image)')
    p.add_argument('--mean-weight', type=float, default=0.0, help='Teacher-free: pull per-channel spatial mean of x to 0')
    p.add_argument('--std-weight', type=float, default=0.0, help='Teacher-free: pull per-channel spatial std of x to 1')
    p.add_argument('--project-input', action='store_true', help='After each step, project x to spatial mean=0 std=1 per channel')
    p.add_argument('--l1-weight', type=float, default=0.0, help='Gray-edge regularization weight')
    p.add_argument('--loss-p', type=float, default=1.0, help='L_p exponent for batch reduction of per-sample losses')
    p.add_argument('--subset-size', type=int, default=5000, help='Number of samples to take before filtering and sorting')
    p.add_argument('--threshold', type=float, default=None, help='Threshold for binarizing target_feat (values >= threshold become 1, else 0)')
    p.add_argument('--select-best-n', type=int, default=inf, help='Select N samples with minimal total_loss to save at the end (default: save all)')
    p.add_argument('--nrow', type=int, default=None, help='Grid row size for saved images (0=auto)')
    p.add_argument('--out', default='output', help='Output directory')
    args = p.parse_args()

    # Parse comma-separated class IDs
    try:
        class_ids = [int(cid.strip()) for cid in args.class_id.split(',')]
    except ValueError:
        raise ValueError(f"Invalid class-id format: '{args.class_id}'. Expected comma-separated integers or '-1' for all classes.")


    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ImageNet normalization constants
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    #mean_awb = mean.mean(dim=1, keepdim=True) #force white balance output

    # 1. Load the inversion model. Torchvision downloads weights on first use.
    print(f"Model: {args.model}")
    model = load_regnet_model(args.model)
    features = model.trunk_output


    # before
    print("ReLUs:", sum(1 for m in model.modules() if isinstance(m, nn.ReLU)))
    print("Softplus:", sum(1 for m in model.modules() if isinstance(m, nn.Softplus)))

    # swap them all, setting beta=2.0 (for instance)
    replace_relu_with_softplus(model, beta=args.beta)

    # after
    print("ReLUs:", sum(1 for m in model.modules() if isinstance(m, nn.ReLU)))
    print("Softplus:", sum(1 for m in model.modules() if isinstance(m, nn.Softplus)))

    model = model.to(device).eval()


    # 1b. Register hooks: target, features, and all BatchNorm outputs
    activation = {}
    hooks = {}

    # Hook trunk_output (pre-pool) features
    def hook_fn(module, input, output):
        activation['feat'] = output
    hooks['feat'] = features.register_forward_hook(hook_fn)

    # 2. Prepare ImageNet validation subset for given class
    # standard ImageNet val preprocessing
    transform_raw = transforms.Compose([
        transforms.Resize(256),          # shorter side → 256
        transforms.CenterCrop(224),      # then take 224×224 center crop
        transforms.ToTensor(),           # [0..1] float tensor
    ])
    
    imgset = ImageNet(root=args.data_dir, split='val', transform=transform_raw)
    # Create string representation for file naming
    # imgset.targets is a plain Python list of length N with the class idx for each sample
    if len(class_ids) == 1 and class_ids[0] == -1:
        class_id_str = 'all'
        indices = list(range(len(imgset.targets)))  # All classes
    else:
        class_id_str = '_'.join(f'{cid:04d}' for cid in sorted(class_ids))
        indices = [i for i, t in enumerate(imgset.targets) if t in class_ids]
    
    # Debug: print class distribution
    class_counts = {}
    for idx in indices:
        class_label = imgset.targets[idx]
        class_counts[class_label] = class_counts.get(class_label, 0) + 1
    print(f"Found {len(indices)} images from classes {sorted(class_ids)}")
    for cid in sorted(class_ids):
        print(f"  Class {cid}: {class_counts.get(cid, 0)} images")

    # Teacher / validation model: pretrained ReLU (Softplus beta -> inf), never args.beta
    model_cls = load_regnet_model(args.model).to(device).eval()
    teacher_activation = {}

    def teacher_hook_fn(module, input, output):
        teacher_activation['feat'] = output

    teacher_hooks = {
        'feat': model_cls.trunk_output.register_forward_hook(teacher_hook_fn),
    }

    sorted_indices = filter_and_sort_by_confidence(
        imgset, indices, model_cls, device, args.batch_size, mean, std, args.subset_size
    )
    
    # Debug: print class distribution after filtering
    if len(class_ids) > 1 or (len(class_ids) == 1 and class_ids[0] != -1):
        class_counts_after = {}
        for idx in sorted_indices:
            class_label = imgset.targets[idx]
            class_counts_after[class_label] = class_counts_after.get(class_label, 0) + 1
        print(f"After filtering: {len(sorted_indices)} correctly classified images")
        for cid in sorted(class_ids):
            print(f"  Class {cid}: {class_counts_after.get(cid, 0)} correctly classified images")
    
    #sorted_indices = indices

    # Balance sampling across classes if multiple classes requested
    if len(class_ids) > 1 and class_ids[0] != -1:
        sorted_indices = balance_classes(sorted_indices, imgset.targets, class_ids, max_count=args.batch_size)
        print(f"Balanced sampling: selected {len(sorted_indices)} images across classes {sorted(class_ids)}")
    
    # Create new loader from sorted images without shuffle
    subset = Subset(imgset, sorted_indices)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=16)

    imgs, _ = next(iter(loader))  # (B,3,224,224)
    imgs = imgs.to(device)
    utils.save_image(imgs, os.path.join(args.out, f'orig_{class_id_str}.png'), nrow=int(sqrt(len(imgs))))

    # 3. Normalize for model input
    imgs_mean = imgs.mean(dim=[1,2,3], keepdim=True)    
    imgs_norm = (imgs - mean) / std

    # 4. Extract and fix target features (teacher = ReLU model_cls, not softplus inversion model)
    tv_loss_orig = tv_loss(imgs_norm)  # (B,) TV of originals for hinge threshold
    with torch.no_grad():
        _ = model_cls(imgs_norm)
        target_feat = teacher_activation['feat'].detach()
        if args.threshold is not None:
            a, b, _ = ab_for_threshold(args.threshold)
            target_feat = b + a * (target_feat >= args.threshold).float()

    # 5. Initialize reconstruction
    noise = torch.randn_like(imgs_norm) * args.sigma
    x = noise.requires_grad_(True)

    opt = optim.Adam([x], lr=args.lr, betas=(0.9, 0.999))
    #opt = optim.SGD([x], lr=args.lr, momentum=0.9)
    #scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: 1 - step / float(args.steps))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    # 6. Reconstruction loop
    for step in range(args.steps+1):
        l2_weight = args.l2_weight# * scheduler.get_last_lr()[0] / args.lr
        opt.zero_grad()
        feat = forward_and_get_feat(model, x, activation)
        (
            loss_kl, loss_mse, loss_mean, loss_std, loss_tv, loss_l1,
            centering_loss, border_loss, total_loss, _,
        ) = compute_all_losses(
            feat, target_feat, x, args.tv_weight,
            l2_weight=l2_weight, mean_weight=args.mean_weight, std_weight=args.std_weight,
            l1_weight=args.l1_weight, tv_loss_orig=tv_loss_orig, batch_norm_p=args.loss_p,
        )

        if step % 50 == 0:
            current_lr = opt.param_groups[0]['lr']
            print(
                f"Step {step}/{args.steps}, MSE: {loss_mse.item():.4f}, Mean: {loss_mean.item():.4f}, "
                f"Std: {loss_std.item():.4f}, KL: {loss_kl.item():.4f}, TV: {loss_tv.item():.4f}, "
                f"L1: {loss_l1.item():.4f}, Centering: {centering_loss.item():.4f}, "
                f"Border: {border_loss.item():.4f}, Total: {total_loss.item():.4f}, LR: {current_lr:.6f}"
            )
            save_reconstructed_images(x, mean, std, os.path.join(args.out, f'recon_{step:04d}.png'), nrow=args.nrow)

        total_loss.backward()
        opt.step()
        scheduler.step()
        if args.project_input:
            with torch.no_grad():
                project_input_spatial(x)

    # Check classification of all reconstructed images with original model
    with torch.no_grad():
        feat = forward_and_get_feat(model, x, activation)
        
        # Get true class labels for all samples
        all_true_labels = torch.tensor([imgset.targets[sorted_indices[i]] for i in range(len(x))], device=device)
        
        # x is already in normalized space, classify directly
        logits_recon = model_cls(x)
        preds_recon = logits_recon.argmax(dim=1)
        correct_mask = (preds_recon == all_true_labels)
        correct_recon = correct_mask.sum().item()
        correct_indices = torch.where(correct_mask)[0].cpu().tolist()
        
    
    # Print classification statistics
    print(f"\nClassification accuracy (all samples):")
    print(f"  Reconstructed images: {correct_recon}/{len(all_true_labels)} ({100*correct_recon/len(all_true_labels):.1f}%)")
    
    # Print per-class accuracy if multiple classes
    if len(class_ids) > 1 and class_ids[0] != -1:
        print(f"\nPer-class accuracy (reconstructed, all samples):")
        for cid in sorted(class_ids):
            class_mask = (all_true_labels == cid)
            if class_mask.sum() > 0:
                class_correct = (preds_recon[class_mask] == all_true_labels[class_mask]).sum().item()
                class_total = class_mask.sum().item()
                print(f"  Class {cid}: {class_correct}/{class_total} ({100*class_correct/class_total:.1f}%)")

    # 7. Select best samples and save (only from correctly classified)
    with torch.no_grad():
        feat = forward_and_get_feat(model, x, activation)
        
        # Compute per-sample losses
        # Calculate per-sample TV loss for originals
        tv_loss_orig_per_sample = tv_loss(imgs_norm)
        _, _, _, _, _, _, _, _, _, loss_kl_per_sample = compute_all_losses(
            feat, target_feat, x, args.tv_weight,
            l2_weight=args.l2_weight, mean_weight=args.mean_weight, std_weight=args.std_weight,
            l1_weight=args.l1_weight, tv_loss_orig=tv_loss_orig_per_sample, batch_norm_p=args.loss_p,
        )
        
        loss_per_sample = loss_kl_per_sample
        
        # Use all samples (do not drop misclassified)
        loss_per_sample_correct = loss_per_sample
        num_correct = loss_per_sample_correct.size(0)
        select_n = min(args.select_best_n, num_correct) if args.select_best_n is not None else num_correct
        
        # Get class labels for all batch indices
        batch_targets_correct = [imgset.targets[sorted_indices[i]] for i in range(num_correct)]
        
        # Select best samples per class if multiple classes requested
        # Use position indices (0 to num_correct-1) for select_best_per_class, then map back to batch indices
        if len(class_ids) > 1 and class_ids[0] != -1:
            position_indices = list(range(num_correct))
            best_position_indices = select_best_per_class(
                position_indices, loss_per_sample_correct, batch_targets_correct, class_ids, select_n
            )
            # Map back to actual batch indices
            best_indices = best_position_indices
            print(f"Selected {len(best_indices)} balanced samples across classes {sorted(class_ids)}")
        else:
            if select_n < num_correct:
                # Use full sort (ascending) for deterministic top-k selection
                _, sorted_idx = torch.sort(loss_per_sample_correct)
                best_indices = sorted_idx[:select_n].cpu().tolist()
                print(f"Selected {select_n} samples with minimal loss (out of {num_correct})")
            else:
                _, sorted_idx = torch.sort(loss_kl_per_sample)
                best_indices = sorted_idx.cpu().tolist()
                print(f"Saving all {num_correct} samples (sorted by KL loss)")
        
        # Select best samples
        x_best = x[best_indices]
        imgs_best = imgs[best_indices]
        #imgs_mean_best = imgs_mean[best_indices]
        
        # Save reconstructed and original images for the best subset
        save_best_images(x_best, imgs_best, mean, std, args.out, class_id_str, nrow=args.nrow)
        
        # Print loss statistics
        print(f"\nBest samples total_loss range: [{loss_per_sample[best_indices].min().item():.4f}, {loss_per_sample[best_indices].max().item():.4f}]")
        print(f"Mean total_loss for best samples: {loss_per_sample[best_indices].mean().item():.4f}")

    for h in hooks.values():
        h.remove()
    for h in teacher_hooks.values():
        h.remove()

    print('Done.')

if __name__ == '__main__':
    main()
