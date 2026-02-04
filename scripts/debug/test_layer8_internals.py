#!/usr/bin/env python3
"""Diagnose WHERE inside layer 8 the HF vs OLMo-core divergence starts.

Hooks into sub-modules of layers 7 (matching) and 8 (diverging) to capture
intermediate values: after q_proj, after QK norm, after RoPE, after attention,
after o_proj, after attention_norm, after residual, after FFN, after FFN_norm.
"""

import logging

import torch
import transformers

from olmo_core.nn.hf.convert import convert_state_from_hf

from open_instruct import logger_utils
from open_instruct.olmo_core_utils import get_transformer_config

logger = logger_utils.setup_logger(__name__)


def report_diff(name, hf_val, olmo_val):
    if hf_val.shape != olmo_val.shape:
        logger.info(f"  {name}: SHAPE MISMATCH hf={hf_val.shape} olmo={olmo_val.shape}")
        return
    diff = (hf_val.float() - olmo_val.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    status = "MATCH" if max_diff == 0 else "DIFF"
    logger.info(f"  {name}: max={max_diff:.6e} mean={mean_diff:.6e} [{status}]")


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
    model_name = "allenai/OLMo-2-0425-1B"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading HF model...")
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)
    hf_model.eval()
    logger.info(f"HF attn_implementation: {hf_model.config._attn_implementation}")

    logger.info("Loading OLMo-core model...")
    hf_config = transformers.AutoConfig.from_pretrained(model_name)
    olmo_config = get_transformer_config(
        model_name, hf_config.vocab_size, attn_backend="torch"
    )
    olmo_model = olmo_config.build(init_device="cpu")
    hf_state = hf_model.state_dict()
    converted_state = convert_state_from_hf(hf_config, hf_state, model_type="olmo2")
    converted_state_gpu = {k: v.to(device) for k, v in converted_state.items()}
    olmo_model = olmo_model.to(device)
    olmo_model.load_state_dict(converted_state_gpu, assign=True, strict=False)
    olmo_model.eval()

    torch.manual_seed(42)
    input_ids = torch.randint(1, 100352, (1, 20), device=device)

    logger.info("\n=== PART 1: RoPE cos/sin comparison ===")
    hf_rope = hf_model.model.rotary_emb
    position_ids = torch.arange(20, device=device).unsqueeze(0)
    hf_embeds = hf_model.model.embed_tokens(input_ids)
    hf_cos, hf_sin = hf_rope(hf_embeds, position_ids)
    logger.info(f"HF cos shape: {hf_cos.shape}, dtype: {hf_cos.dtype}")
    logger.info(f"HF sin shape: {hf_sin.shape}, dtype: {hf_sin.dtype}")

    olmo_rope = olmo_model.blocks["0"].attention.rope
    logger.info(f"OLMo-core RoPE type: {type(olmo_rope).__name__}")
    logger.info(f"OLMo-core full_precision: {olmo_rope.full_precision}")
    olmo_sin_full, olmo_cos_full = olmo_rope._get_rotary_embedding(20, device)
    olmo_cos = olmo_cos_full[:20, :]
    olmo_sin = olmo_sin_full[:20, :]
    logger.info(f"OLMo cos shape: {olmo_cos.shape}, dtype: {olmo_cos.dtype}")
    logger.info(f"OLMo sin shape: {olmo_sin.shape}, dtype: {olmo_sin.dtype}")

    hf_cos_squeezed = hf_cos.squeeze(0)
    hf_sin_squeezed = hf_sin.squeeze(0)
    cos_diff = (hf_cos_squeezed.float() - olmo_cos.float()).abs()
    sin_diff = (hf_sin_squeezed.float() - olmo_sin.float()).abs()
    logger.info(f"cos diff: max={cos_diff.max().item():.6e} mean={cos_diff.mean().item():.6e}")
    logger.info(f"sin diff: max={sin_diff.max().item():.6e} mean={sin_diff.mean().item():.6e}")

    for block_key in ["0", "7", "8", "15"]:
        block_rope = olmo_model.blocks[block_key].attention.rope
        block_sin, block_cos = block_rope._get_rotary_embedding(20, device)
        same_cos = torch.equal(block_cos, olmo_cos_full)
        same_sin = torch.equal(block_sin, olmo_sin_full)
        logger.info(f"Block {block_key} RoPE same as block 0: cos={same_cos} sin={same_sin}")

    logger.info("\n=== PART 2: Attention mask check ===")
    from transformers.masking_utils import create_causal_mask

    causal_mask = create_causal_mask(
        config=hf_model.config,
        input_embeds=hf_embeds,
        attention_mask=None,
        cache_position=torch.arange(20, device=device),
        past_key_values=None,
    )
    if causal_mask is None:
        logger.info("HF causal_mask is None (FA2/no mask mode)")
    else:
        logger.info(f"HF causal_mask shape: {causal_mask.shape}, dtype: {causal_mask.dtype}")
        logger.info(f"HF causal_mask unique values: {causal_mask.unique().tolist()}")

    logger.info("\n=== PART 3: Sub-module hooks inside layers 7 and 8 ===")
    for target_layer in [7, 8]:
        logger.info(f"\n--- Layer {target_layer} internals ---")
        hf_layer = hf_model.model.layers[target_layer]
        olmo_block = olmo_model.blocks[str(target_layer)]

        hf_captures = {}
        olmo_captures = {}
        handles = []

        def make_hf_hook(name):
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    hf_captures[name] = out[0].detach().clone()
                else:
                    hf_captures[name] = out.detach().clone()
            return hook

        def make_olmo_hook(name):
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    olmo_captures[name] = out[0].detach().clone()
                else:
                    olmo_captures[name] = out.detach().clone()
            return hook

        def make_hf_pre_hook(name):
            def hook(module, inp):
                hf_captures[name] = inp[0].detach().clone()
            return hook

        def make_olmo_pre_hook(name):
            def hook(module, inp):
                olmo_captures[name] = inp[0].detach().clone()
            return hook

        handles.append(hf_layer.register_forward_pre_hook(make_hf_pre_hook("layer_input")))
        handles.append(olmo_block.register_forward_pre_hook(make_olmo_pre_hook("layer_input")))

        handles.append(hf_layer.self_attn.register_forward_hook(make_hf_hook("attn_output")))
        handles.append(olmo_block.attention.register_forward_hook(make_olmo_hook("attn_output")))

        handles.append(
            hf_layer.post_attention_layernorm.register_forward_hook(
                make_hf_hook("attn_norm_output")
            )
        )
        handles.append(
            olmo_block.attention_norm.register_forward_hook(
                make_olmo_hook("attn_norm_output")
            )
        )

        handles.append(hf_layer.mlp.register_forward_hook(make_hf_hook("ffn_output")))
        handles.append(olmo_block.feed_forward.register_forward_hook(make_olmo_hook("ffn_output")))

        handles.append(
            hf_layer.post_feedforward_layernorm.register_forward_hook(
                make_hf_hook("ffn_norm_output")
            )
        )
        handles.append(
            olmo_block.feed_forward_norm.register_forward_hook(
                make_olmo_hook("ffn_norm_output")
            )
        )

        handles.append(hf_layer.register_forward_hook(make_hf_hook("layer_output")))
        handles.append(olmo_block.register_forward_hook(make_olmo_hook("layer_output")))

        handles.append(hf_layer.self_attn.q_proj.register_forward_hook(make_hf_hook("q_proj")))
        handles.append(olmo_block.attention.w_q.register_forward_hook(make_olmo_hook("q_proj")))
        handles.append(hf_layer.self_attn.k_proj.register_forward_hook(make_hf_hook("k_proj")))
        handles.append(olmo_block.attention.w_k.register_forward_hook(make_olmo_hook("k_proj")))
        handles.append(hf_layer.self_attn.v_proj.register_forward_hook(make_hf_hook("v_proj")))
        handles.append(olmo_block.attention.w_v.register_forward_hook(make_olmo_hook("v_proj")))
        handles.append(hf_layer.self_attn.o_proj.register_forward_hook(make_hf_hook("o_proj")))
        handles.append(olmo_block.attention.w_out.register_forward_hook(make_olmo_hook("o_proj")))

        handles.append(hf_layer.self_attn.q_norm.register_forward_hook(make_hf_hook("q_norm")))
        handles.append(olmo_block.attention.q_norm.register_forward_hook(make_olmo_hook("q_norm")))
        handles.append(hf_layer.self_attn.k_norm.register_forward_hook(make_hf_hook("k_norm")))
        handles.append(olmo_block.attention.k_norm.register_forward_hook(make_olmo_hook("k_norm")))

        with torch.no_grad():
            _ = hf_model(input_ids)
            _ = olmo_model(input_ids)

        for h in handles:
            h.remove()

        comparison_order = [
            "layer_input",
            "q_proj",
            "k_proj",
            "v_proj",
            "q_norm",
            "k_norm",
            "attn_output",
            "attn_norm_output",
            "ffn_output",
            "ffn_norm_output",
            "layer_output",
        ]
        for name in comparison_order:
            if name in hf_captures and name in olmo_captures:
                report_diff(name, hf_captures[name], olmo_captures[name])
            else:
                missing_in = []
                if name not in hf_captures:
                    missing_in.append("HF")
                if name not in olmo_captures:
                    missing_in.append("OLMo")
                logger.info(f"  {name}: MISSING in {', '.join(missing_in)}")

    logger.info("\n=== PART 4: Isolate attention sub-computation at layer 8 ===")
    logger.info("Running layer 8's attention with identical input manually...")
    hf_layer8 = hf_model.model.layers[8]
    olmo_block8 = olmo_model.blocks["8"]

    hf_block_outputs = []
    olmo_block_outputs = []
    handles2 = []

    def capture_layer7_hf(module, inp, out):
        hf_block_outputs.append(out[0].detach().clone() if isinstance(out, tuple) else out.detach().clone())

    def capture_layer7_olmo(module, inp, out):
        olmo_block_outputs.append(out.detach().clone())

    handles2.append(hf_model.model.layers[7].register_forward_hook(capture_layer7_hf))
    handles2.append(olmo_model.blocks["7"].register_forward_hook(capture_layer7_olmo))

    with torch.no_grad():
        _ = hf_model(input_ids)
        _ = olmo_model(input_ids)

    for h in handles2:
        h.remove()

    layer8_input = hf_block_outputs[0]
    logger.info(f"Layer 8 input (from layer 7 output): shape={layer8_input.shape}")
    logger.info(
        f"Layer 7 output diff: {(hf_block_outputs[0] - olmo_block_outputs[0]).abs().max().item():.6e}"
    )

    hf_q_out = hf_layer8.self_attn.q_norm(hf_layer8.self_attn.q_proj(layer8_input))
    olmo_q_out = olmo_block8.attention.q_norm(olmo_block8.attention.w_q(layer8_input))
    report_diff("manual_q_after_norm", hf_q_out, olmo_q_out)

    hf_k_out = hf_layer8.self_attn.k_norm(hf_layer8.self_attn.k_proj(layer8_input))
    olmo_k_out = olmo_block8.attention.k_norm(olmo_block8.attention.w_k(layer8_input))
    report_diff("manual_k_after_norm", hf_k_out, olmo_k_out)

    hf_v_out = hf_layer8.self_attn.v_proj(layer8_input)
    olmo_v_out = olmo_block8.attention.w_v(layer8_input)
    report_diff("manual_v", hf_v_out, olmo_v_out)

    B, T, _ = layer8_input.shape
    head_dim = hf_layer8.self_attn.head_dim
    n_heads = hf_model.config.num_attention_heads

    hf_q_reshaped = hf_q_out.view(B, T, n_heads, head_dim).transpose(1, 2)
    hf_k_reshaped = hf_k_out.view(B, T, n_heads, head_dim).transpose(1, 2)

    olmo_q_reshaped = olmo_q_out.view(B, T, -1, head_dim)
    olmo_k_reshaped = olmo_k_out.view(B, T, -1, head_dim)

    hf_cos, hf_sin = hf_model.model.rotary_emb(layer8_input, position_ids)
    from transformers.models.olmo2.modeling_olmo2 import apply_rotary_pos_emb

    hf_q_roped, hf_k_roped = apply_rotary_pos_emb(hf_q_reshaped, hf_k_reshaped, hf_cos, hf_sin)

    olmo_rope8 = olmo_block8.attention.rope
    olmo_q_roped, olmo_k_roped = olmo_rope8(
        olmo_q_reshaped, olmo_k_reshaped, head_first=False
    )

    hf_q_roped_compare = hf_q_roped.transpose(1, 2)
    hf_k_roped_compare = hf_k_roped.transpose(1, 2)
    report_diff("manual_q_after_rope", hf_q_roped_compare, olmo_q_roped)
    report_diff("manual_k_after_rope", hf_k_roped_compare, olmo_k_roped)

    logger.info("\n=== PART 5: Check o_proj input ===")
    hf_o_in = []
    olmo_o_in = []
    handles3 = []

    def capture_o_proj_input_hf(module, inp):
        hf_o_in.append(inp[0].detach().clone())

    def capture_o_proj_input_olmo(module, inp):
        olmo_o_in.append(inp[0].detach().clone())

    handles3.append(
        hf_layer8.self_attn.o_proj.register_forward_pre_hook(capture_o_proj_input_hf)
    )
    handles3.append(
        olmo_block8.attention.w_out.register_forward_pre_hook(capture_o_proj_input_olmo)
    )

    with torch.no_grad():
        _ = hf_model(input_ids)
        _ = olmo_model(input_ids)

    for h in handles3:
        h.remove()

    if hf_o_in and olmo_o_in:
        logger.info(f"HF o_proj input shape: {hf_o_in[0].shape}")
        logger.info(f"OLMo w_out input shape: {olmo_o_in[0].shape}")
        report_diff("o_proj_input", hf_o_in[0], olmo_o_in[0])


if __name__ == "__main__":
    main()
