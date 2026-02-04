#!/usr/bin/env python3
"""Test whether tensor memory layout (contiguous vs transposed) affects FA2 output.

This script tests the hypothesis that the numerical differences between OLMo-core
and HuggingFace arise from different tensor memory layouts when calling FA2:
- OLMo-core: q.view(B, T, n_heads, head_dim) -> contiguous
- HuggingFace: q.view(B, n_heads, T, head_dim).transpose(1, 2) -> non-contiguous

Both have the same logical shape [B, T, n_heads, head_dim] and same values,
but different memory strides.
"""

import logging

import torch
import transformers
from olmo_core.nn.hf.convert import convert_state_from_hf

from open_instruct import logger_utils
from open_instruct.olmo_core_utils import get_transformer_config

logger = logger_utils.setup_logger(__name__)


def test_fa2_contiguity_basic():
    """Test FA2 output with contiguous vs non-contiguous inputs."""
    import flash_attn

    B, T, n_heads, head_dim = 1, 20, 16, 128
    device = torch.device("cuda")
    dtype = torch.bfloat16

    torch.manual_seed(42)
    q_flat = torch.randn(B, T, n_heads * head_dim, device=device, dtype=dtype)
    k_flat = torch.randn(B, T, n_heads * head_dim, device=device, dtype=dtype)
    v_flat = torch.randn(B, T, n_heads * head_dim, device=device, dtype=dtype)

    q_contiguous = q_flat.view(B, T, n_heads, head_dim)
    k_contiguous = k_flat.view(B, T, n_heads, head_dim)
    v_contiguous = v_flat.view(B, T, n_heads, head_dim)

    q_transposed = q_flat.view(B, T, n_heads, head_dim).transpose(1, 2).transpose(1, 2)
    k_transposed = k_flat.view(B, T, n_heads, head_dim).transpose(1, 2).transpose(1, 2)
    v_transposed = v_flat.view(B, T, n_heads, head_dim).transpose(1, 2).transpose(1, 2)

    q_hf_style = q_flat.view(B, n_heads, T, head_dim).transpose(1, 2)
    k_hf_style = k_flat.view(B, n_heads, T, head_dim).transpose(1, 2)
    v_hf_style = v_flat.view(B, n_heads, T, head_dim).transpose(1, 2)

    logger.info("=== TENSOR LAYOUT COMPARISON ===")
    logger.info(f"Contiguous strides: {q_contiguous.stride()}, is_contiguous: {q_contiguous.is_contiguous()}")
    logger.info(f"Double-transposed strides: {q_transposed.stride()}, is_contiguous: {q_transposed.is_contiguous()}")
    logger.info(f"HF-style strides: {q_hf_style.stride()}, is_contiguous: {q_hf_style.is_contiguous()}")

    logger.info(f"\nValues match (contiguous vs HF-style): {torch.equal(q_contiguous, q_hf_style)}")

    logger.info("\n=== FA2 OUTPUT COMPARISON ===")

    with torch.no_grad():
        out_contiguous = flash_attn.flash_attn_func(q_contiguous, k_contiguous, v_contiguous, causal=True)
        out_transposed = flash_attn.flash_attn_func(q_transposed, k_transposed, v_transposed, causal=True)
        out_hf_style = flash_attn.flash_attn_func(q_hf_style, k_hf_style, v_hf_style, causal=True)

    diff_transposed = (out_contiguous - out_transposed).abs()
    diff_hf = (out_contiguous - out_hf_style).abs()

    logger.info(f"Contiguous vs double-transposed: max_diff={diff_transposed.max().item():.6e}, mean_diff={diff_transposed.mean().item():.6e}")
    logger.info(f"Contiguous vs HF-style:          max_diff={diff_hf.max().item():.6e}, mean_diff={diff_hf.mean().item():.6e}")

    out_contiguous_2 = flash_attn.flash_attn_func(q_contiguous, k_contiguous, v_contiguous, causal=True)
    self_diff = (out_contiguous - out_contiguous_2).abs()
    logger.info(f"Self-consistency:                 max_diff={self_diff.max().item():.6e}")


def test_fa2_contiguity_sweep():
    """Test across different sequence lengths."""
    import flash_attn

    B, n_heads, head_dim = 1, 16, 128
    device = torch.device("cuda")
    dtype = torch.bfloat16

    logger.info("\n=== SEQUENCE LENGTH SWEEP ===")
    for seq_len in [1, 2, 4, 8, 10, 16, 20, 32, 64, 128, 256, 512]:
        torch.manual_seed(42)
        q_flat = torch.randn(B, seq_len, n_heads * head_dim, device=device, dtype=dtype)
        k_flat = torch.randn(B, seq_len, n_heads * head_dim, device=device, dtype=dtype)
        v_flat = torch.randn(B, seq_len, n_heads * head_dim, device=device, dtype=dtype)

        q_contiguous = q_flat.view(B, seq_len, n_heads, head_dim)
        k_contiguous = k_flat.view(B, seq_len, n_heads, head_dim)
        v_contiguous = v_flat.view(B, seq_len, n_heads, head_dim)

        q_hf = q_flat.view(B, n_heads, seq_len, head_dim).transpose(1, 2)
        k_hf = k_flat.view(B, n_heads, seq_len, head_dim).transpose(1, 2)
        v_hf = v_flat.view(B, n_heads, seq_len, head_dim).transpose(1, 2)

        with torch.no_grad():
            out_c = flash_attn.flash_attn_func(q_contiguous, k_contiguous, v_contiguous, causal=True)
            out_h = flash_attn.flash_attn_func(q_hf, k_hf, v_hf, causal=True)

        diff = (out_c - out_h).abs()
        status = "MATCH" if diff.max().item() == 0 else "DIFF"
        logger.info(f"  seq_len={seq_len:4d}: max_diff={diff.max().item():.6e} [{status}]")


def test_full_model_with_contiguous_fix():
    """Test if making HF tensors contiguous before FA2 eliminates model differences."""
    model_name = "allenai/OLMo-2-0425-1B"
    device = torch.device("cuda")

    logger.info("\n=== FULL MODEL COMPARISON ===")
    logger.info(f"Loading models on {device}...")

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    hf_model.eval()

    hf_config = transformers.AutoConfig.from_pretrained(model_name)
    olmo_config = get_transformer_config(model_name, hf_config.vocab_size, attn_backend="flash_2")
    olmo_model = olmo_config.build(init_device="cpu")

    hf_state = hf_model.state_dict()
    converted_state = convert_state_from_hf(hf_config, hf_state, model_type="olmo2")
    converted_state_gpu = {k: v.to(device) for k, v in converted_state.items()}
    olmo_model = olmo_model.to(device)
    olmo_model.load_state_dict(converted_state_gpu, assign=True, strict=False)
    olmo_model.eval()

    torch.manual_seed(42)
    input_ids = torch.randint(1, 100352, (1, 20), device=device)

    logger.info("\n--- Test A: Standard comparison (original behavior) ---")
    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits
        olmo_logits = olmo_model(input_ids)
    diff = (hf_logits - olmo_logits).abs()
    logger.info(f"Max diff: {diff.max().item():.6e}")
    logger.info(f"Mean diff: {diff.mean().item():.6e}")

    logger.info("\n--- Test B: Layer-by-layer hidden state comparison ---")
    hf_states = []
    olmo_states = []

    def hf_hook(module, inp, out):
        hf_states.append(out[0].detach().clone())

    def olmo_hook(module, inp, out):
        olmo_states.append(out.detach().clone())

    hf_handles = [layer.register_forward_hook(hf_hook) for layer in hf_model.model.layers]
    olmo_block_keys = list(olmo_model.blocks.keys())
    olmo_handles = [olmo_model.blocks[key].register_forward_hook(olmo_hook) for key in olmo_block_keys]

    with torch.no_grad():
        _ = hf_model(input_ids)
        _ = olmo_model(input_ids)

    for h in hf_handles + olmo_handles:
        h.remove()

    for i in range(min(len(hf_states), len(olmo_states))):
        d = (hf_states[i] - olmo_states[i]).abs()
        status = "MATCH" if d.max().item() == 0 else "DIFF"
        logger.info(f"Layer {i:2d}: max_diff={d.max().item():.6e}, mean_diff={d.mean().item():.6e} [{status}]")

    logger.info("\n--- Test C: Inspect attention module tensor strides ---")
    hf_strides = {}
    olmo_strides = {}

    def hf_attn_hook(name):
        def hook(module, inp, out):
            q_proj_out = module.q_proj(inp[0])
            normed = module.q_norm(q_proj_out)
            hidden_shape = (*inp[0].shape[:-1], -1, module.head_dim)
            q_after_reshape = normed.view(hidden_shape).transpose(1, 2)
            q_for_fa2 = q_after_reshape.transpose(1, 2)
            hf_strides[name] = {
                "shape": q_for_fa2.shape,
                "stride": q_for_fa2.stride(),
                "is_contiguous": q_for_fa2.is_contiguous(),
            }
        return hook

    def olmo_attn_hook(name):
        def hook(module, inp, out):
            x = inp[0]
            B, T = x.shape[:2]
            q = module.w_q(x)
            if module.q_norm is not None:
                q = module.q_norm(q)
            q = q.view(B, T, -1, module.head_dim)
            olmo_strides[name] = {
                "shape": q.shape,
                "stride": q.stride(),
                "is_contiguous": q.is_contiguous(),
            }
        return hook

    hf_handles = [
        hf_model.model.layers[i].self_attn.register_forward_hook(hf_attn_hook(f"layer_{i}"))
        for i in [0, 7, 8]
    ]
    olmo_handles = [
        olmo_model.blocks[key].attention.register_forward_hook(olmo_attn_hook(f"block_{key}"))
        for key in ["0", "7", "8"]
    ]

    with torch.no_grad():
        _ = hf_model(input_ids)
        _ = olmo_model(input_ids)

    for h in hf_handles + olmo_handles:
        h.remove()

    for name in sorted(hf_strides.keys()):
        info = hf_strides[name]
        logger.info(f"HF {name}: shape={info['shape']}, stride={info['stride']}, contiguous={info['is_contiguous']}")

    for name in sorted(olmo_strides.keys()):
        info = olmo_strides[name]
        logger.info(f"OLMo {name}: shape={info['shape']}, stride={info['stride']}, contiguous={info['is_contiguous']}")

    logger.info("\n--- Test D: Compare with torch SDPA backend (should show 0 diff) ---")
    hf_model_sdpa = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    hf_model_sdpa.eval()

    olmo_config_torch = get_transformer_config(model_name, hf_config.vocab_size, attn_backend="torch")
    olmo_model_torch = olmo_config_torch.build(init_device="cpu")
    converted_state_torch = convert_state_from_hf(hf_config, hf_model_sdpa.state_dict(), model_type="olmo2")
    converted_state_torch_gpu = {k: v.to(device) for k, v in converted_state_torch.items()}
    olmo_model_torch = olmo_model_torch.to(device)
    olmo_model_torch.load_state_dict(converted_state_torch_gpu, assign=True, strict=False)
    olmo_model_torch.eval()

    with torch.no_grad():
        hf_logits_sdpa = hf_model_sdpa(input_ids).logits
        olmo_logits_torch = olmo_model_torch(input_ids)
    diff_sdpa = (hf_logits_sdpa - olmo_logits_torch).abs()
    logger.info(f"SDPA/torch max diff: {diff_sdpa.max().item():.6e}")
    logger.info(f"SDPA/torch mean diff: {diff_sdpa.mean().item():.6e}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
    test_fa2_contiguity_basic()
    test_fa2_contiguity_sweep()
    test_full_model_with_contiguous_fix()


if __name__ == "__main__":
    main()
