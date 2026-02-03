#!/usr/bin/env python3
"""Compare OLMo-core and HuggingFace models layer by layer.

This script identifies where numerical differences originate between the two
implementations by comparing outputs at each layer.
"""

import torch
import transformers
from olmo_core.nn.hf.convert import convert_state_from_hf
from olmo_core.nn.transformer import TransformerConfig

from open_instruct.olmo_core_utils import get_transformer_config


def main():
    model_name = "allenai/OLMo-2-0425-1B"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading models on {device}...")

    # Load HuggingFace model
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    hf_model.eval()

    # Load OLMo-core model
    hf_config = transformers.AutoConfig.from_pretrained(model_name)
    olmo_config = get_transformer_config(model_name, hf_config.vocab_size, attn_backend="flash_2")
    olmo_model = olmo_config.build(init_device="cpu")

    # Convert and load weights
    hf_state = hf_model.state_dict()
    converted_state = convert_state_from_hf(hf_config, hf_state, model_type="olmo2")
    converted_state_gpu = {k: v.to(device) for k, v in converted_state.items()}
    olmo_model = olmo_model.to(device)
    olmo_model.load_state_dict(converted_state_gpu, assign=True, strict=False)
    olmo_model.eval()

    # Create test input
    torch.manual_seed(42)
    seq_len = 20
    input_ids = torch.randint(1, 100352, (1, seq_len), device=device)
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Input tokens: {input_ids[0, :10].tolist()}...")

    # Compare embeddings
    print("\n" + "=" * 60)
    print("EMBEDDING COMPARISON")
    print("=" * 60)

    with torch.no_grad():
        hf_embeds = hf_model.model.embed_tokens(input_ids)
        olmo_embeds = olmo_model.embeddings(input_ids)

    embed_diff = (hf_embeds - olmo_embeds).abs()
    print(f"HF embeddings shape: {hf_embeds.shape}")
    print(f"OLMo embeddings shape: {olmo_embeds.shape}")
    print(f"Max diff: {embed_diff.max().item():.6e}")
    print(f"Mean diff: {embed_diff.mean().item():.6e}")

    # Compare layer by layer
    print("\n" + "=" * 60)
    print("LAYER-BY-LAYER COMPARISON")
    print("=" * 60)

    # Get intermediate outputs from HuggingFace
    hf_hidden = hf_embeds
    olmo_hidden = olmo_embeds

    num_layers = min(len(hf_model.model.layers), len(olmo_model.blocks))

    for layer_idx in range(num_layers):
        hf_layer = hf_model.model.layers[layer_idx]
        olmo_block = olmo_model.blocks[layer_idx]

        with torch.no_grad():
            # HuggingFace layer forward
            hf_out = hf_layer(
                hf_hidden,
                position_ids=torch.arange(seq_len, device=device).unsqueeze(0),
            )
            hf_hidden_new = hf_out[0]

            # OLMo-core block forward (need to handle differently)
            # OLMo-core uses a different interface
            olmo_hidden_new = olmo_block(olmo_hidden)

        layer_diff = (hf_hidden_new - olmo_hidden_new).abs()
        max_diff = layer_diff.max().item()
        mean_diff = layer_diff.mean().item()

        # Check if difference is significant
        status = "✓" if max_diff < 1e-3 else "✗"
        print(f"Layer {layer_idx:2d}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e} {status}")

        if max_diff > 0.01:
            # Drill down into this layer
            print(f"  → Large diff detected, investigating...")

            # Compare attention outputs
            with torch.no_grad():
                # HF attention
                hf_normed = hf_layer.input_layernorm(hf_hidden)
                olmo_normed = olmo_block.attention_norm(olmo_hidden)

                norm_diff = (hf_normed - olmo_normed).abs().max().item()
                print(f"  → After attention norm: max_diff={norm_diff:.6e}")

        hf_hidden = hf_hidden_new
        olmo_hidden = olmo_hidden_new

    # Compare final outputs
    print("\n" + "=" * 60)
    print("FINAL OUTPUT COMPARISON")
    print("=" * 60)

    with torch.no_grad():
        # Final layer norm
        hf_final_norm = hf_model.model.norm(hf_hidden)
        olmo_final_norm = olmo_model.ln_f(olmo_hidden)

        norm_diff = (hf_final_norm - olmo_final_norm).abs()
        print(f"After final norm: max_diff={norm_diff.max().item():.6e}")

        # LM head
        hf_logits = hf_model.lm_head(hf_final_norm)
        olmo_logits = olmo_model.lm_head(olmo_final_norm)

        logits_diff = (hf_logits - olmo_logits).abs()
        print(f"Final logits: max_diff={logits_diff.max().item():.6e}")

        # Full forward pass comparison
        hf_full_logits = hf_model(input_ids).logits
        olmo_full_logits = olmo_model(input_ids)

        full_diff = (hf_full_logits - olmo_full_logits).abs()
        print(f"\nFull forward pass: max_diff={full_diff.max().item():.6e}")
        print(f"Full forward pass: mean_diff={full_diff.mean().item():.6e}")

        # Sample logits
        print(f"\nSample logits at position 5:")
        print(f"  HF:    {hf_full_logits[0, 5, :5].tolist()}")
        print(f"  OLMo:  {olmo_full_logits[0, 5, :5].tolist()}")


if __name__ == "__main__":
    main()
