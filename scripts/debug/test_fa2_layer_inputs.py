#!/usr/bin/env python3
"""Deep diagnostic: capture exact FA2 inputs/outputs at each layer.

This script hooks INSIDE the attention modules of both OLMo-core and HuggingFace
to capture the exact q, k, v tensors BEFORE the FA2 call and the attention output
AFTER the FA2 call. This will definitively determine whether the numerical
differences arise from different inputs to FA2 or from FA2 itself.
"""

import logging

import torch
import transformers
from olmo_core.nn.hf.convert import convert_state_from_hf

from open_instruct import logger_utils
from open_instruct.olmo_core_utils import get_transformer_config

logger = logger_utils.setup_logger(__name__)


def capture_fa2_inputs():
    model_name = "allenai/OLMo-2-0425-1B"
    device = torch.device("cuda")

    logger.info("Loading HuggingFace model (FA2)...")
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    hf_model.eval()

    logger.info("Loading OLMo-core model (FA2)...")
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

    logger.info("=== PART 1: Capture block/layer outputs ===")
    hf_layer_outputs = []
    olmo_layer_outputs = []

    def hf_layer_hook(module, inp, out):
        hf_layer_outputs.append(out[0].detach().clone())

    def olmo_layer_hook(module, inp, out):
        olmo_layer_outputs.append(out.detach().clone())

    hf_handles = [layer.register_forward_hook(hf_layer_hook) for layer in hf_model.model.layers]
    olmo_keys = list(olmo_model.blocks.keys())
    olmo_handles = [olmo_model.blocks[k].register_forward_hook(olmo_layer_hook) for k in olmo_keys]

    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits
        olmo_logits = olmo_model(input_ids)

    for h in hf_handles + olmo_handles:
        h.remove()

    first_diff_layer = None
    for i in range(len(hf_layer_outputs)):
        diff = (hf_layer_outputs[i] - olmo_layer_outputs[i]).abs()
        max_diff = diff.max().item()
        status = "MATCH" if max_diff == 0 else "DIFF"
        logger.info(f"Layer {i:2d} output: max_diff={max_diff:.6e} [{status}]")
        if max_diff > 0 and first_diff_layer is None:
            first_diff_layer = i

    logits_diff = (hf_logits - olmo_logits).abs()
    logger.info(f"Final logits: max_diff={logits_diff.max().item():.6e}")

    if first_diff_layer is None:
        logger.info("All layers match exactly! No FA2 differences detected.")
        return

    logger.info(f"\nFirst divergence at layer {first_diff_layer}")

    logger.info(f"\n=== PART 2: Deep dive into layer {first_diff_layer} ===")
    logger.info("Capturing attention inputs (q, k, v after RoPE) and outputs")

    hf_attn_data = {}
    olmo_attn_data = {}

    def make_hf_flash_hook(layer_idx):
        """Hook the flash_attention_forward function call inside HF attention."""
        import transformers.integrations.flash_attention as fa_module

        original_fn = fa_module.flash_attention_forward

        def patched_fn(module, query, key, value, attention_mask, **kwargs):
            q_for_fa2 = query.transpose(1, 2)
            k_for_fa2 = key.transpose(1, 2)
            v_for_fa2 = value.transpose(1, 2)
            hf_attn_data[layer_idx] = {
                "q_before_transpose": query.detach().clone(),
                "k_before_transpose": key.detach().clone(),
                "v_before_transpose": value.detach().clone(),
                "q_for_fa2": q_for_fa2.detach().clone(),
                "k_for_fa2": k_for_fa2.detach().clone(),
                "v_for_fa2": v_for_fa2.detach().clone(),
                "q_stride": q_for_fa2.stride(),
                "k_stride": k_for_fa2.stride(),
                "v_stride": v_for_fa2.stride(),
                "q_contiguous": q_for_fa2.is_contiguous(),
                "scaling": kwargs.get("scaling", None),
            }
            result = original_fn(module, query, key, value, attention_mask, **kwargs)
            hf_attn_data[layer_idx]["attn_output"] = result[0].detach().clone()
            return result

        return patched_fn, original_fn

    def make_olmo_sdpa_hook(layer_idx):
        """Hook the FA2 dispatch inside OLMo-core attention."""
        def hook(module, inp, out):
            olmo_attn_data[layer_idx] = {
                "attn_output": out.detach().clone(),
            }
        return hook

    target_layers = list(range(min(first_diff_layer + 2, 16)))

    for layer_idx in target_layers:
        olmo_attn = olmo_model.blocks[str(layer_idx)].attention
        olmo_attn._debug_layer_idx = layer_idx
        olmo_attn._debug_data = olmo_attn_data

        original_sdpa = olmo_attn.sdpa

        class WrappedSDPA:
            def __init__(self, sdpa, layer_idx, data_dict):
                self._sdpa = sdpa
                self._layer_idx = layer_idx
                self._data = data_dict

            def __call__(self, q, k, v, **kwargs):
                self._data[self._layer_idx] = {
                    "q_for_fa2": q.detach().clone(),
                    "k_for_fa2": k.detach().clone(),
                    "v_for_fa2": v.detach().clone(),
                    "q_stride": q.stride(),
                    "k_stride": k.stride(),
                    "v_stride": v.stride(),
                    "q_contiguous": q.is_contiguous(),
                }
                result = self._sdpa(q, k, v, **kwargs)
                self._data[self._layer_idx]["attn_output"] = result.detach().clone()
                return result

            def __getattr__(self, name):
                return getattr(self._sdpa, name)

        olmo_attn.sdpa = WrappedSDPA(original_sdpa, layer_idx, olmo_attn_data)

    import transformers.integrations.flash_attention as fa_module

    original_flash_fn = fa_module.flash_attention_forward
    call_count = [0]

    def global_flash_hook(module, query, key, value, attention_mask, **kwargs):
        layer_idx = call_count[0]
        call_count[0] += 1

        if layer_idx in target_layers:
            q_for_fa2 = query.transpose(1, 2)
            k_for_fa2 = key.transpose(1, 2)
            v_for_fa2 = value.transpose(1, 2)
            hf_attn_data[layer_idx] = {
                "q_for_fa2": q_for_fa2.detach().clone(),
                "k_for_fa2": k_for_fa2.detach().clone(),
                "v_for_fa2": v_for_fa2.detach().clone(),
                "q_stride": q_for_fa2.stride(),
                "k_stride": k_for_fa2.stride(),
                "v_stride": v_for_fa2.stride(),
                "q_contiguous": q_for_fa2.is_contiguous(),
                "scaling": kwargs.get("scaling", None),
            }

        result = original_flash_fn(module, query, key, value, attention_mask, **kwargs)

        if layer_idx in target_layers:
            hf_attn_data[layer_idx]["attn_output"] = result[0].detach().clone()

        return result

    fa_module.flash_attention_forward = global_flash_hook

    with torch.no_grad():
        _ = hf_model(input_ids)

    fa_module.flash_attention_forward = original_flash_fn

    call_count[0] = 0
    with torch.no_grad():
        _ = olmo_model(input_ids)

    for layer_idx in target_layers:
        olmo_attn = olmo_model.blocks[str(layer_idx)].attention
        if hasattr(olmo_attn, '_orig_sdpa'):
            olmo_attn.sdpa = olmo_attn._orig_sdpa

    logger.info("\n=== FA2 INPUT COMPARISON (q, k, v before FA2 call) ===")
    for layer_idx in target_layers:
        if layer_idx not in hf_attn_data or layer_idx not in olmo_attn_data:
            logger.info(f"Layer {layer_idx}: MISSING DATA (hf={layer_idx in hf_attn_data}, olmo={layer_idx in olmo_attn_data})")
            continue

        hf_d = hf_attn_data[layer_idx]
        olmo_d = olmo_attn_data[layer_idx]

        q_diff = (hf_d["q_for_fa2"] - olmo_d["q_for_fa2"]).abs()
        k_diff = (hf_d["k_for_fa2"] - olmo_d["k_for_fa2"]).abs()
        v_diff = (hf_d["v_for_fa2"] - olmo_d["v_for_fa2"]).abs()

        q_status = "MATCH" if q_diff.max().item() == 0 else "DIFF"
        k_status = "MATCH" if k_diff.max().item() == 0 else "DIFF"
        v_status = "MATCH" if v_diff.max().item() == 0 else "DIFF"

        logger.info(
            f"Layer {layer_idx:2d} FA2 inputs: "
            f"q_diff={q_diff.max().item():.6e} [{q_status}], "
            f"k_diff={k_diff.max().item():.6e} [{k_status}], "
            f"v_diff={v_diff.max().item():.6e} [{v_status}]"
        )
        logger.info(
            f"  HF  strides: q={hf_d['q_stride']}, contiguous={hf_d['q_contiguous']}"
        )
        logger.info(
            f"  OLMo strides: q={olmo_d['q_stride']}, contiguous={olmo_d['q_contiguous']}"
        )

    logger.info("\n=== FA2 OUTPUT COMPARISON ===")
    for layer_idx in target_layers:
        if layer_idx not in hf_attn_data or layer_idx not in olmo_attn_data:
            continue

        hf_out = hf_attn_data[layer_idx]["attn_output"]
        olmo_out = olmo_attn_data[layer_idx]["attn_output"]

        if hf_out.shape != olmo_out.shape:
            logger.info(f"Layer {layer_idx}: shape mismatch HF={hf_out.shape} vs OLMo={olmo_out.shape}")
            if hf_out.dim() == 4 and olmo_out.dim() == 4:
                if hf_out.shape[1] != olmo_out.shape[1]:
                    hf_out = hf_out.transpose(1, 2)

        out_diff = (hf_out - olmo_out).abs()
        out_status = "MATCH" if out_diff.max().item() == 0 else "DIFF"
        logger.info(
            f"Layer {layer_idx:2d} FA2 output: "
            f"max_diff={out_diff.max().item():.6e}, "
            f"mean_diff={out_diff.mean().item():.6e} [{out_status}]"
        )

    logger.info("\n=== PART 3: Direct FA2 call with captured tensors ===")
    logger.info("Calling FA2 directly with OLMo-captured q,k,v vs HF-captured q,k,v")

    import flash_attn

    for layer_idx in target_layers:
        if layer_idx not in hf_attn_data or layer_idx not in olmo_attn_data:
            continue

        hf_d = hf_attn_data[layer_idx]
        olmo_d = olmo_attn_data[layer_idx]

        hf_q = hf_d["q_for_fa2"].contiguous()
        hf_k = hf_d["k_for_fa2"].contiguous()
        hf_v = hf_d["v_for_fa2"].contiguous()

        olmo_q = olmo_d["q_for_fa2"].contiguous()
        olmo_k = olmo_d["k_for_fa2"].contiguous()
        olmo_v = olmo_d["v_for_fa2"].contiguous()

        with torch.no_grad():
            out_from_hf = flash_attn.flash_attn_func(hf_q, hf_k, hf_v, causal=True)
            out_from_olmo = flash_attn.flash_attn_func(olmo_q, olmo_k, olmo_v, causal=True)

        direct_diff = (out_from_hf - out_from_olmo).abs()
        status = "MATCH" if direct_diff.max().item() == 0 else "DIFF"
        logger.info(
            f"Layer {layer_idx:2d} direct FA2: "
            f"max_diff={direct_diff.max().item():.6e} [{status}]"
        )

        if (hf_d["q_for_fa2"] - olmo_d["q_for_fa2"]).abs().max().item() == 0:
            out_hf_orig_stride = flash_attn.flash_attn_func(
                hf_d["q_for_fa2"], hf_d["k_for_fa2"], hf_d["v_for_fa2"], causal=True
            )
            out_olmo_orig_stride = flash_attn.flash_attn_func(
                olmo_d["q_for_fa2"], olmo_d["k_for_fa2"], olmo_d["v_for_fa2"], causal=True
            )
            stride_diff = (out_hf_orig_stride - out_olmo_orig_stride).abs()
            logger.info(
                f"Layer {layer_idx:2d} stride test: "
                f"max_diff={stride_diff.max().item():.6e} "
                f"(same data, different strides)"
            )

    logger.info("\n=== PART 4: Compare with SDPA ===")
    logger.info("Loading models with SDPA/torch backend...")

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

    sdpa_layer_outputs_hf = []
    sdpa_layer_outputs_olmo = []

    def hf_sdpa_hook(module, inp, out):
        sdpa_layer_outputs_hf.append(out[0].detach().clone())

    def olmo_sdpa_hook(module, inp, out):
        sdpa_layer_outputs_olmo.append(out.detach().clone())

    hf_handles = [layer.register_forward_hook(hf_sdpa_hook) for layer in hf_model_sdpa.model.layers]
    olmo_handles = [olmo_model_torch.blocks[k].register_forward_hook(olmo_sdpa_hook) for k in olmo_keys]

    with torch.no_grad():
        hf_logits_sdpa = hf_model_sdpa(input_ids).logits
        olmo_logits_torch = olmo_model_torch(input_ids)

    for h in hf_handles + olmo_handles:
        h.remove()

    logger.info("\nSDPA layer-by-layer comparison:")
    for i in range(len(sdpa_layer_outputs_hf)):
        diff = (sdpa_layer_outputs_hf[i] - sdpa_layer_outputs_olmo[i]).abs()
        max_diff = diff.max().item()
        status = "MATCH" if max_diff == 0 else "DIFF"
        logger.info(f"Layer {i:2d} output (SDPA): max_diff={max_diff:.6e} [{status}]")

    sdpa_logits_diff = (hf_logits_sdpa - olmo_logits_torch).abs()
    logger.info(f"SDPA logits: max_diff={sdpa_logits_diff.max().item():.6e}")

    logger.info("\n=== PART 5: Sequence length sweep with FA2 ===")
    for seq_len in [1, 2, 4, 8, 10, 16, 20, 32, 64]:
        torch.manual_seed(42)
        test_input = torch.randint(1, 100352, (1, seq_len), device=device)
        with torch.no_grad():
            hf_out = hf_model(test_input).logits
            olmo_out = olmo_model(test_input)
        diff = (hf_out - olmo_out).abs().max().item()
        status = "MATCH" if diff == 0 else "DIFF"
        logger.info(f"seq_len={seq_len:3d}: max_logits_diff={diff:.6e} [{status}]")


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
    capture_fa2_inputs()


if __name__ == "__main__":
    main()
