#!/usr/bin/env python3
"""Compare weights between HF and OLMo-core models layer by layer.

Identifies any weight mismatches that could explain why layers 0-7 match
but layer 8+ diverges.
"""

import logging

import torch
import transformers
from olmo_core.nn.hf.convert import convert_state_from_hf

from open_instruct import logger_utils
from open_instruct.olmo_core_utils import get_transformer_config

logger = logger_utils.setup_logger(__name__)


def compare_weights():
    model_name = "allenai/OLMo-2-0425-1B"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading HuggingFace model...")
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
    ).to(device)
    hf_model.eval()

    logger.info("Loading OLMo-core model...")
    hf_config = transformers.AutoConfig.from_pretrained(model_name)
    olmo_config = get_transformer_config(model_name, hf_config.vocab_size, attn_backend="flash_2")
    olmo_model = olmo_config.build(init_device="cpu")

    hf_state = hf_model.state_dict()
    converted_state = convert_state_from_hf(hf_config, hf_state, model_type="olmo2")
    converted_state_gpu = {k: v.to(device) for k, v in converted_state.items()}

    logger.info("\n=== PART 1: Check converted state keys ===")
    olmo_model_keys = set(olmo_model.state_dict().keys())
    converted_keys = set(converted_state.keys())

    missing_in_converted = olmo_model_keys - converted_keys
    extra_in_converted = converted_keys - olmo_model_keys

    if missing_in_converted:
        logger.info(f"MISSING keys (in OLMo model but NOT in converted state): {len(missing_in_converted)}")
        for k in sorted(missing_in_converted):
            logger.info(f"  MISSING: {k}")
    else:
        logger.info("No missing keys - all OLMo model params are in converted state")

    if extra_in_converted:
        logger.info(f"EXTRA keys (in converted state but NOT in OLMo model): {len(extra_in_converted)}")
        for k in sorted(extra_in_converted):
            logger.info(f"  EXTRA: {k}")
    else:
        logger.info("No extra keys in converted state")

    olmo_model = olmo_model.to(device)
    load_result = olmo_model.load_state_dict(converted_state_gpu, assign=True, strict=False)
    logger.info(f"\nload_state_dict result:")
    logger.info(f"  missing_keys: {load_result.missing_keys}")
    logger.info(f"  unexpected_keys: {load_result.unexpected_keys}")

    olmo_model.eval()

    logger.info("\n=== PART 2: Direct weight comparison at each layer ===")
    weight_pairs = [
        ("self_attn.q_proj.weight", "attention.w_q.weight"),
        ("self_attn.k_proj.weight", "attention.w_k.weight"),
        ("self_attn.v_proj.weight", "attention.w_v.weight"),
        ("self_attn.o_proj.weight", "attention.w_out.weight"),
        ("self_attn.q_norm.weight", "attention.q_norm.weight"),
        ("self_attn.k_norm.weight", "attention.k_norm.weight"),
        ("mlp.gate_proj.weight", "feed_forward.w1.weight"),
        ("mlp.down_proj.weight", "feed_forward.w2.weight"),
        ("mlp.up_proj.weight", "feed_forward.w3.weight"),
        ("post_attention_layernorm.weight", "attention_norm.weight"),
        ("post_feedforward_layernorm.weight", "feed_forward_norm.weight"),
    ]

    for layer_idx in range(16):
        hf_layer = hf_model.model.layers[layer_idx]
        olmo_block = olmo_model.blocks[str(layer_idx)]

        layer_ok = True
        for hf_name, olmo_name in weight_pairs:
            hf_param = hf_layer
            for part in hf_name.split("."):
                hf_param = getattr(hf_param, part)

            olmo_param = olmo_block
            for part in olmo_name.split("."):
                olmo_param = getattr(olmo_param, part)

            if hf_param.shape != olmo_param.shape:
                logger.info(f"Layer {layer_idx:2d} {hf_name}: SHAPE MISMATCH hf={hf_param.shape} olmo={olmo_param.shape}")
                layer_ok = False
                continue

            diff = (hf_param - olmo_param).abs()
            if diff.max().item() > 0:
                logger.info(
                    f"Layer {layer_idx:2d} {hf_name}: WEIGHT DIFF max={diff.max().item():.6e} mean={diff.mean().item():.6e}"
                )
                layer_ok = False

        status = "OK" if layer_ok else "MISMATCH"
        logger.info(f"Layer {layer_idx:2d}: {status}")

    logger.info("\n=== PART 3: Check embeddings and LM head ===")
    embed_diff = (hf_model.model.embed_tokens.weight - olmo_model.embeddings.weight).abs()
    logger.info(f"Embeddings: max_diff={embed_diff.max().item():.6e}")

    final_norm_diff = (hf_model.model.norm.weight - olmo_model.lm_head.norm.weight).abs()
    logger.info(f"Final norm: max_diff={final_norm_diff.max().item():.6e}")

    lm_head_diff = (hf_model.lm_head.weight - olmo_model.lm_head.w_out.weight).abs()
    logger.info(f"LM head: max_diff={lm_head_diff.max().item():.6e}")

    logger.info("\n=== PART 4: HF state dict key inventory ===")
    hf_keys_by_layer = {}
    for k in sorted(hf_state.keys()):
        if k.startswith("model.layers."):
            parts = k.split(".")
            layer_num = int(parts[2])
            param_name = ".".join(parts[3:])
            if layer_num not in hf_keys_by_layer:
                hf_keys_by_layer[layer_num] = []
            hf_keys_by_layer[layer_num].append(param_name)

    logger.info(f"Layer 7 HF keys: {sorted(hf_keys_by_layer.get(7, []))}")
    logger.info(f"Layer 8 HF keys: {sorted(hf_keys_by_layer.get(8, []))}")

    keys_7 = set(hf_keys_by_layer.get(7, []))
    keys_8 = set(hf_keys_by_layer.get(8, []))
    if keys_7 != keys_8:
        logger.info(f"DIFFERENT KEYS between layer 7 and 8!")
        logger.info(f"  Only in 7: {keys_7 - keys_8}")
        logger.info(f"  Only in 8: {keys_8 - keys_7}")
    else:
        logger.info("Layer 7 and 8 have identical key structure")

    logger.info("\n=== PART 5: Forward pass with weight verification ===")
    torch.manual_seed(42)
    input_ids = torch.randint(1, 100352, (1, 20), device=device)

    hf_hidden = []
    olmo_hidden = []
    hf_attn_in = []
    olmo_attn_in = []

    def hf_layer_pre_hook(layer_idx):
        def hook(module, inp):
            hf_attn_in.append(inp[0].detach().clone())
        return hook

    def olmo_layer_pre_hook(layer_idx):
        def hook(module, inp):
            olmo_attn_in.append(inp[0].detach().clone())
        return hook

    def hf_layer_hook(module, inp, out):
        hf_hidden.append(out[0].detach().clone())

    def olmo_layer_hook(module, inp, out):
        olmo_hidden.append(out.detach().clone())

    handles = []
    for i, layer in enumerate(hf_model.model.layers):
        handles.append(layer.register_forward_pre_hook(hf_layer_pre_hook(i)))
        handles.append(layer.register_forward_hook(hf_layer_hook))
    olmo_keys = list(olmo_model.blocks.keys())
    for i, key in enumerate(olmo_keys):
        handles.append(olmo_model.blocks[key].register_forward_pre_hook(olmo_layer_pre_hook(i)))
        handles.append(olmo_model.blocks[key].register_forward_hook(olmo_layer_hook))

    with torch.no_grad():
        _ = hf_model(input_ids)
        _ = olmo_model(input_ids)

    for h in handles:
        h.remove()

    logger.info("\nLayer INPUT comparison (hidden states entering each layer):")
    for i in range(len(hf_attn_in)):
        diff = (hf_attn_in[i] - olmo_attn_in[i]).abs()
        max_diff = diff.max().item()
        status = "MATCH" if max_diff == 0 else "DIFF"
        logger.info(f"Layer {i:2d} input: max_diff={max_diff:.6e} [{status}]")

    logger.info("\nLayer OUTPUT comparison:")
    for i in range(len(hf_hidden)):
        diff = (hf_hidden[i] - olmo_hidden[i]).abs()
        max_diff = diff.max().item()
        status = "MATCH" if max_diff == 0 else "DIFF"
        logger.info(f"Layer {i:2d} output: max_diff={max_diff:.6e} [{status}]")


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
    compare_weights()


if __name__ == "__main__":
    main()
