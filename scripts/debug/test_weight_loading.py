#!/usr/bin/env python3
"""Check weight loading: verify all weights match between HF and OLMo-core."""

import logging

import torch
import transformers
from olmo_core.nn.hf.convert import convert_state_from_hf

from open_instruct import logger_utils
from open_instruct.olmo_core_utils import get_transformer_config

logger = logger_utils.setup_logger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
    model_name = "allenai/OLMo-2-0425-1B"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)
    hf_model.eval()

    hf_config = transformers.AutoConfig.from_pretrained(model_name)
    olmo_config = get_transformer_config(model_name, hf_config.vocab_size, attn_backend="flash_2")
    olmo_model = olmo_config.build(init_device="cpu")

    hf_state = hf_model.state_dict()
    converted_state = convert_state_from_hf(hf_config, hf_state, model_type="olmo2")

    logger.info("=== WEIGHT CONVERSION DIAGNOSTICS ===")
    logger.info(f"HF state dict keys: {len(hf_state)}")
    logger.info(f"Converted state dict keys: {len(converted_state)}")

    olmo_state = olmo_model.state_dict()
    logger.info(f"OLMo model state dict keys: {len(olmo_state)}")

    converted_keys = set(converted_state.keys())
    olmo_keys = set(olmo_state.keys())

    missing_in_converted = olmo_keys - converted_keys
    extra_in_converted = converted_keys - olmo_keys

    if missing_in_converted:
        logger.info(f"\nKeys in OLMo model but MISSING from converted state ({len(missing_in_converted)}):")
        for k in sorted(missing_in_converted):
            logger.info(f"  MISSING: {k}")
    else:
        logger.info("\nAll OLMo model keys present in converted state.")

    if extra_in_converted:
        logger.info(f"\nExtra keys in converted state not in OLMo model ({len(extra_in_converted)}):")
        for k in sorted(extra_in_converted):
            logger.info(f"  EXTRA: {k}")

    converted_state_gpu = {k: v.to(device) for k, v in converted_state.items()}
    olmo_model = olmo_model.to(device)
    load_result = olmo_model.load_state_dict(converted_state_gpu, assign=True, strict=False)
    logger.info(f"\nload_state_dict result:")
    logger.info(f"  missing_keys: {load_result.missing_keys}")
    logger.info(f"  unexpected_keys: {load_result.unexpected_keys}")

    logger.info("\n=== PER-LAYER WEIGHT COMPARISON ===")
    logger.info("Comparing converted weights vs OLMo model weights after loading:")

    olmo_loaded_state = olmo_model.state_dict()
    for key in sorted(olmo_loaded_state.keys()):
        if key in converted_state_gpu:
            diff = (olmo_loaded_state[key].float() - converted_state_gpu[key].float()).abs().max().item()
            if diff > 0:
                logger.info(f"  {key}: max_diff={diff:.6e} MISMATCH")
        else:
            logger.info(f"  {key}: NOT IN CONVERTED STATE")

    logger.info("\n=== DIRECT WEIGHT COMPARISON: HF vs OLMo-core ===")
    logger.info("Checking if weights match for each layer component:")

    for layer_idx in range(16):
        diffs = {}

        hf_q = hf_state[f"model.layers.{layer_idx}.self_attn.q_proj.weight"]
        olmo_q = olmo_loaded_state[f"blocks.{layer_idx}.attention.w_q.weight"]
        diffs["q_proj"] = (hf_q.float() - olmo_q.float()).abs().max().item()

        hf_k = hf_state[f"model.layers.{layer_idx}.self_attn.k_proj.weight"]
        olmo_k = olmo_loaded_state[f"blocks.{layer_idx}.attention.w_k.weight"]
        diffs["k_proj"] = (hf_k.float() - olmo_k.float()).abs().max().item()

        hf_v = hf_state[f"model.layers.{layer_idx}.self_attn.v_proj.weight"]
        olmo_v = olmo_loaded_state[f"blocks.{layer_idx}.attention.w_v.weight"]
        diffs["v_proj"] = (hf_v.float() - olmo_v.float()).abs().max().item()

        hf_o = hf_state[f"model.layers.{layer_idx}.self_attn.o_proj.weight"]
        olmo_o = olmo_loaded_state[f"blocks.{layer_idx}.attention.w_out.weight"]
        diffs["o_proj"] = (hf_o.float() - olmo_o.float()).abs().max().item()

        hf_qn = hf_state[f"model.layers.{layer_idx}.self_attn.q_norm.weight"]
        olmo_qn = olmo_loaded_state[f"blocks.{layer_idx}.attention.q_norm.weight"]
        diffs["q_norm"] = (hf_qn.float() - olmo_qn.float()).abs().max().item()

        hf_kn = hf_state[f"model.layers.{layer_idx}.self_attn.k_norm.weight"]
        olmo_kn = olmo_loaded_state[f"blocks.{layer_idx}.attention.k_norm.weight"]
        diffs["k_norm"] = (hf_kn.float() - olmo_kn.float()).abs().max().item()

        hf_gate = hf_state[f"model.layers.{layer_idx}.mlp.gate_proj.weight"]
        olmo_w1 = olmo_loaded_state[f"blocks.{layer_idx}.feed_forward.w1.weight"]
        diffs["gate_proj"] = (hf_gate.float() - olmo_w1.float()).abs().max().item()

        hf_down = hf_state[f"model.layers.{layer_idx}.mlp.down_proj.weight"]
        olmo_w2 = olmo_loaded_state[f"blocks.{layer_idx}.feed_forward.w2.weight"]
        diffs["down_proj"] = (hf_down.float() - olmo_w2.float()).abs().max().item()

        hf_up = hf_state[f"model.layers.{layer_idx}.mlp.up_proj.weight"]
        olmo_w3 = olmo_loaded_state[f"blocks.{layer_idx}.feed_forward.w3.weight"]
        diffs["up_proj"] = (hf_up.float() - olmo_w3.float()).abs().max().item()

        any_mismatch = any(v > 0 for v in diffs.values())
        status = "MISMATCH" if any_mismatch else "OK"
        logger.info(f"Layer {layer_idx:2d}: {status}")
        if any_mismatch:
            for name, diff in diffs.items():
                if diff > 0:
                    logger.info(f"  {name}: {diff:.6e}")

    logger.info("\n=== NORM WEIGHT MAPPING CHECK ===")
    logger.info("HF OLMo2 uses POST-norm (ReorderedNormTransformerBlock).")
    logger.info("Checking norm weight mapping:")

    for layer_idx in range(16):
        hf_input_ln = hf_state.get(f"model.layers.{layer_idx}.input_layernorm.weight")
        hf_post_attn_ln = hf_state[f"model.layers.{layer_idx}.post_attention_layernorm.weight"]
        hf_post_ffn_ln = hf_state[f"model.layers.{layer_idx}.post_feedforward_layernorm.weight"]

        olmo_attn_norm = olmo_loaded_state[f"blocks.{layer_idx}.attention_norm.weight"]
        olmo_ffn_norm = olmo_loaded_state[f"blocks.{layer_idx}.feed_forward_norm.weight"]

        if hf_input_ln is not None:
            logger.info(f"Layer {layer_idx}: HAS input_layernorm (unexpected for OLMo2!)")

        attn_norm_diff = (hf_post_attn_ln.float() - olmo_attn_norm.float()).abs().max().item()
        ffn_norm_diff = (hf_post_ffn_ln.float() - olmo_ffn_norm.float()).abs().max().item()

        if attn_norm_diff > 0 or ffn_norm_diff > 0:
            logger.info(
                f"Layer {layer_idx:2d} norms: "
                f"post_attn_ln→attn_norm diff={attn_norm_diff:.6e}, "
                f"post_ffn_ln→ffn_norm diff={ffn_norm_diff:.6e}"
            )

    logger.info("\n=== CHECK: input_layernorm vs post_attention_layernorm ===")
    logger.info("OLMo2 has BOTH input_layernorm and post_attention_layernorm.")
    logger.info("The convert maps post_attention_layernorm → attention_norm.")
    logger.info("But what about input_layernorm? Does OLMo-core use it?")

    for layer_idx in range(16):
        hf_input_ln_key = f"model.layers.{layer_idx}.input_layernorm.weight"
        hf_post_attn_key = f"model.layers.{layer_idx}.post_attention_layernorm.weight"

        if hf_input_ln_key in hf_state:
            hf_input = hf_state[hf_input_ln_key]
            hf_post = hf_state[hf_post_attn_key]
            same = torch.equal(hf_input, hf_post)
            diff = (hf_input.float() - hf_post.float()).abs().max().item()
            logger.info(
                f"Layer {layer_idx:2d}: input_layernorm == post_attention_layernorm? "
                f"{same} (max_diff={diff:.6e})"
            )


if __name__ == "__main__":
    main()
