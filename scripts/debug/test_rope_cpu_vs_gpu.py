#!/usr/bin/env python3
"""Verify that RoPE inv_freq differs between CPU and GPU computation."""

import logging

import torch

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    theta = 500000
    dim = 128

    logger.info("=== Test 1: inv_freq on CPU vs GPU ===")
    arange_cpu = torch.arange(0, dim, 2, dtype=torch.float, device="cpu")
    inv_freq_cpu = 1.0 / (theta ** (arange_cpu / dim))

    arange_gpu = torch.arange(0, dim, 2, dtype=torch.float, device=device)
    inv_freq_gpu = 1.0 / (theta ** (arange_gpu / dim))

    inv_freq_diff = (inv_freq_cpu - inv_freq_gpu.cpu()).abs()
    logger.info(f"inv_freq max diff: {inv_freq_diff.max().item():.6e}")
    logger.info(f"inv_freq mean diff: {inv_freq_diff.mean().item():.6e}")
    logger.info(f"Number of differing elements: {(inv_freq_diff > 0).sum().item()}/{inv_freq_diff.numel()}")

    nonzero_indices = torch.where(inv_freq_diff > 0)[0]
    if len(nonzero_indices) > 0:
        logger.info(f"First 10 differing indices: {nonzero_indices[:10].tolist()}")
        for idx in nonzero_indices[:5]:
            logger.info(
                f"  idx={idx.item()}: CPU={inv_freq_cpu[idx].item():.15e}, "
                f"GPU={inv_freq_gpu[idx].cpu().item():.15e}, "
                f"diff={inv_freq_diff[idx].item():.6e}"
            )

    logger.info("\n=== Test 2: Full cos/sin comparison ===")
    seq_len = 20

    seq_cpu = torch.arange(seq_len, dtype=torch.float, device="cpu")
    freqs_cpu = torch.einsum("i , j -> i j", seq_cpu, inv_freq_cpu)
    positions_cpu = torch.cat((freqs_cpu, freqs_cpu), dim=-1)
    cos_cpu = positions_cpu.cos()
    sin_cpu = positions_cpu.sin()

    seq_gpu = torch.arange(seq_len, dtype=torch.float, device=device)
    freqs_gpu = torch.einsum("i , j -> i j", seq_gpu, inv_freq_gpu)
    positions_gpu = torch.cat((freqs_gpu, freqs_gpu), dim=-1)
    cos_gpu = positions_gpu.cos()
    sin_gpu = positions_gpu.sin()

    cos_diff = (cos_cpu - cos_gpu.cpu()).abs()
    sin_diff = (sin_cpu - sin_gpu.cpu()).abs()
    logger.info(f"cos max diff: {cos_diff.max().item():.6e}")
    logger.info(f"sin max diff: {sin_diff.max().item():.6e}")

    logger.info("\n=== Test 3: HF-style vs OLMo-style computation ===")

    hf_inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(
                device="cpu", dtype=torch.float
            )
            / dim
        )
    )
    hf_inv_freq_gpu = hf_inv_freq.to(device)

    inv_freq_expanded = hf_inv_freq_gpu[None, :, None].float()
    position_ids = torch.arange(0, seq_len, device=device)
    position_ids_expanded = position_ids[None, None, :].float()

    with torch.autocast(device_type="cuda", enabled=False):
        hf_freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        hf_emb = torch.cat((hf_freqs, hf_freqs), dim=-1)
        hf_cos = hf_emb.cos()
        hf_sin = hf_emb.sin()

    olmo_inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim)
    )
    olmo_seq = torch.arange(seq_len, device=device, dtype=torch.float)
    olmo_freqs = torch.einsum("i , j -> i j", olmo_seq, olmo_inv_freq)
    olmo_positions = torch.cat((olmo_freqs, olmo_freqs), dim=-1)
    olmo_cos = olmo_positions.cos()
    olmo_sin = olmo_positions.sin()

    hf_cos_2d = hf_cos.squeeze(0)
    cos_comparison = (hf_cos_2d - olmo_cos).abs()
    sin_comparison = (hf_sin.squeeze(0) - olmo_sin).abs()
    logger.info(f"HF vs OLMo cos max diff: {cos_comparison.max().item():.6e}")
    logger.info(f"HF vs OLMo sin max diff: {sin_comparison.max().item():.6e}")

    logger.info("\n=== Test 4: Fix by computing OLMo inv_freq on CPU ===")
    olmo_inv_freq_cpu = 1.0 / (
        theta
        ** (torch.arange(0, dim, 2, dtype=torch.float, device="cpu") / dim)
    )
    olmo_inv_freq_cpu_on_gpu = olmo_inv_freq_cpu.to(device)

    olmo_freqs_fixed = torch.einsum(
        "i , j -> i j", olmo_seq, olmo_inv_freq_cpu_on_gpu
    )
    olmo_positions_fixed = torch.cat((olmo_freqs_fixed, olmo_freqs_fixed), dim=-1)
    olmo_cos_fixed = olmo_positions_fixed.cos()
    olmo_sin_fixed = olmo_positions_fixed.sin()

    cos_fixed_diff = (hf_cos_2d - olmo_cos_fixed).abs()
    sin_fixed_diff = (hf_sin.squeeze(0) - olmo_sin_fixed).abs()
    logger.info(f"HF vs OLMo-fixed cos max diff: {cos_fixed_diff.max().item():.6e}")
    logger.info(f"HF vs OLMo-fixed sin max diff: {sin_fixed_diff.max().item():.6e}")

    inv_freq_fixed_diff = (hf_inv_freq - olmo_inv_freq_cpu).abs()
    logger.info(f"inv_freq diff (both CPU): {inv_freq_fixed_diff.max().item():.6e}")


if __name__ == "__main__":
    main()
