#!/usr/bin/env python3
"""Compare OLMo-core and HuggingFace models layer by layer.

This script identifies where numerical differences originate between the two
implementations by comparing outputs at each layer.
"""

import torch
import transformers
from olmo_core.nn.hf.convert import convert_state_from_hf

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

    # Test 1: Simple forward (no padding)
    print("\n" + "=" * 60)
    print("TEST 1: SIMPLE FORWARD (no padding)")
    print("=" * 60)

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

    # Full forward pass comparison
    print("\n" + "=" * 60)
    print("FULL FORWARD PASS COMPARISON")
    print("=" * 60)

    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits
        olmo_logits = olmo_model(input_ids)

    logits_diff = (hf_logits - olmo_logits).abs()
    print(f"HF logits shape: {hf_logits.shape}")
    print(f"OLMo logits shape: {olmo_logits.shape}")
    print(f"Max diff: {logits_diff.max().item():.6e}")
    print(f"Mean diff: {logits_diff.mean().item():.6e}")

    # Find position with max diff
    max_pos = logits_diff.argmax()
    b, s, v = max_pos // (seq_len * 100352), (max_pos % (seq_len * 100352)) // 100352, max_pos % 100352
    print(f"Max diff at position: batch={b.item()}, seq={s.item()}, vocab={v.item()}")
    print(f"  HF value: {hf_logits[b, s, v].item():.6f}")
    print(f"  OLMo value: {olmo_logits[b, s, v].item():.6f}")

    # Compare per-position
    print("\n" + "=" * 60)
    print("PER-POSITION MAX DIFF")
    print("=" * 60)

    for pos in range(min(10, seq_len)):
        pos_diff = logits_diff[0, pos, :].max().item()
        print(f"Position {pos:2d}: max_diff={pos_diff:.6e}")

    # Sample logits comparison
    print("\n" + "=" * 60)
    print("SAMPLE LOGITS AT POSITION 5")
    print("=" * 60)

    print(f"HF:   {hf_logits[0, 5, :10].tolist()}")
    print(f"OLMo: {olmo_logits[0, 5, :10].tolist()}")

    # Compare intermediate layer outputs using hooks
    print("\n" + "=" * 60)
    print("LAYER-BY-LAYER HIDDEN STATE COMPARISON")
    print("=" * 60)

    hf_hidden_states = []
    olmo_hidden_states = []

    def hf_hook(module, input, output):
        hf_hidden_states.append(output[0].detach())

    def olmo_hook(module, input, output):
        olmo_hidden_states.append(output.detach())

    # Register hooks on each layer
    hf_handles = []
    olmo_handles = []

    for i, layer in enumerate(hf_model.model.layers):
        hf_handles.append(layer.register_forward_hook(hf_hook))

    olmo_block_keys = list(olmo_model.blocks.keys())
    for key in olmo_block_keys:
        olmo_handles.append(olmo_model.blocks[key].register_forward_hook(olmo_hook))

    # Run forward passes with hooks
    with torch.no_grad():
        _ = hf_model(input_ids)
        _ = olmo_model(input_ids)

    # Remove hooks
    for h in hf_handles + olmo_handles:
        h.remove()

    # Compare layer outputs
    num_layers = min(len(hf_hidden_states), len(olmo_hidden_states))
    print(f"\nComparing {num_layers} layers:")

    first_large_diff_layer = None
    for i in range(num_layers):
        diff = (hf_hidden_states[i] - olmo_hidden_states[i]).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        status = "✓" if max_diff < 1e-3 else "✗"
        print(f"Layer {i:2d}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e} {status}")

        if max_diff > 0.01 and first_large_diff_layer is None:
            first_large_diff_layer = i

    if first_large_diff_layer is not None:
        print(f"\nFirst layer with large diff: {first_large_diff_layer}")

        # Investigate this layer more
        print(f"\nInvestigating Layer {first_large_diff_layer}...")

        hf_layer = hf_model.model.layers[first_large_diff_layer]
        olmo_block = olmo_model.blocks[olmo_block_keys[first_large_diff_layer]]

        # Get input to this layer
        if first_large_diff_layer == 0:
            hf_input = hf_embeds
            olmo_input = olmo_embeds
        else:
            hf_input = hf_hidden_states[first_large_diff_layer - 1]
            olmo_input = olmo_hidden_states[first_large_diff_layer - 1]

        with torch.no_grad():
            # Compare layer norms
            hf_normed = hf_layer.input_layernorm(hf_input)
            olmo_normed = olmo_block.attention_norm(olmo_input)
            norm_diff = (hf_normed - olmo_normed).abs().max().item()
            print(f"  Pre-attention norm diff: {norm_diff:.6e}")

            # Compare post-attention norm
            hf_post_norm = hf_layer.post_attention_layernorm(hf_input)
            olmo_post_norm = olmo_block.feed_forward_norm(olmo_input)
            post_norm_diff = (hf_post_norm - olmo_post_norm).abs().max().item()
            print(f"  Pre-FFN norm diff: {post_norm_diff:.6e}")

    # Test 2: Padded input with attention mask
    print("\n" + "=" * 60)
    print("TEST 2: PADDED INPUT WITH ATTENTION MASK")
    print("=" * 60)

    pad_token_id = 0
    content_len = 15
    padded_seq_len = 20
    padding_len = padded_seq_len - content_len

    torch.manual_seed(42)
    content_tokens = torch.randint(1, 100352, (1, content_len), device=device)
    padding_tokens = torch.full((1, padding_len), pad_token_id, device=device)
    padded_input_ids = torch.cat([content_tokens, padding_tokens], dim=1)

    attention_mask = torch.ones(1, padded_seq_len, device=device)
    attention_mask[0, content_len:] = 0

    print(f"Input shape: {padded_input_ids.shape}")
    print(f"Content length: {content_len}, Padding length: {padding_len}")
    print(f"Attention mask: {attention_mask[0].tolist()}")

    with torch.no_grad():
        hf_logits_padded = hf_model(padded_input_ids, attention_mask=attention_mask).logits
        olmo_logits_padded = olmo_model(padded_input_ids)

    logits_diff_content = (hf_logits_padded[0, :content_len] - olmo_logits_padded[0, :content_len]).abs()
    logits_diff_padding = (hf_logits_padded[0, content_len:] - olmo_logits_padded[0, content_len:]).abs()

    print(f"\nContent region (0:{content_len}):")
    print(f"  Max diff: {logits_diff_content.max().item():.6e}")
    print(f"  Mean diff: {logits_diff_content.mean().item():.6e}")

    print(f"\nPadding region ({content_len}:{padded_seq_len}):")
    print(f"  Max diff: {logits_diff_padding.max().item():.6e}")
    print(f"  Mean diff: {logits_diff_padding.mean().item():.6e}")

    print("\nPer-position max diff (content region):")
    for pos in range(content_len):
        pos_diff = logits_diff_content[pos, :].max().item()
        print(f"  Position {pos:2d}: max_diff={pos_diff:.6e}")

    # Test 3: Batched forward (like DPO chosen/rejected)
    print("\n" + "=" * 60)
    print("TEST 3: BATCHED FORWARD (simulating DPO chosen/rejected)")
    print("=" * 60)

    torch.manual_seed(42)
    batch_size = 2
    seq_len_batch = 20
    batched_input_ids = torch.randint(1, 100352, (batch_size, seq_len_batch), device=device)

    print(f"Batch shape: {batched_input_ids.shape}")

    with torch.no_grad():
        hf_logits_batch = hf_model(batched_input_ids).logits
        olmo_logits_batch = olmo_model(batched_input_ids)

    for b in range(batch_size):
        batch_diff = (hf_logits_batch[b] - olmo_logits_batch[b]).abs()
        print(f"\nBatch {b}:")
        print(f"  Max diff: {batch_diff.max().item():.6e}")
        print(f"  Mean diff: {batch_diff.mean().item():.6e}")

    # Test 4: Batched forward with padding (like actual DPO data)
    print("\n" + "=" * 60)
    print("TEST 4: BATCHED FORWARD WITH DIFFERENT PADDING PER SAMPLE")
    print("=" * 60)

    torch.manual_seed(42)
    batch_size = 2
    max_seq_len = 20
    seq_lens = [15, 12]

    batched_padded_ids = torch.full((batch_size, max_seq_len), pad_token_id, device=device)
    batched_attn_mask = torch.zeros(batch_size, max_seq_len, device=device)

    for b, seq_len_b in enumerate(seq_lens):
        batched_padded_ids[b, :seq_len_b] = torch.randint(1, 100352, (seq_len_b,), device=device)
        batched_attn_mask[b, :seq_len_b] = 1

    print(f"Batch shape: {batched_padded_ids.shape}")
    print(f"Sequence lengths: {seq_lens}")
    print(f"Attention mask row sums: {batched_attn_mask.sum(dim=1).tolist()}")

    with torch.no_grad():
        hf_logits_bp = hf_model(batched_padded_ids, attention_mask=batched_attn_mask).logits
        olmo_logits_bp = olmo_model(batched_padded_ids)

    for b, seq_len_b in enumerate(seq_lens):
        content_diff = (hf_logits_bp[b, :seq_len_b] - olmo_logits_bp[b, :seq_len_b]).abs()
        padding_diff = (hf_logits_bp[b, seq_len_b:] - olmo_logits_bp[b, seq_len_b:]).abs()
        print(f"\nBatch {b} (content_len={seq_len_b}):")
        print(f"  Content max diff: {content_diff.max().item():.6e}")
        print(f"  Content mean diff: {content_diff.mean().item():.6e}")
        print(f"  Padding max diff: {padding_diff.max().item():.6e}")
        print(f"  Padding mean diff: {padding_diff.mean().item():.6e}")

    # Test 5: OLMo-core with doc_lens (packed forward)
    print("\n" + "=" * 60)
    print("TEST 5: OLMO-CORE PACKED FORWARD (doc_lens) vs BATCHED")
    print("=" * 60)

    torch.manual_seed(42)
    doc1_len = 10
    doc2_len = 8
    all_tokens = torch.randint(1, 100352, (doc1_len + doc2_len,), device=device)
    doc1_tokens = all_tokens[:doc1_len]
    doc2_tokens = all_tokens[doc1_len:]

    packed_input = torch.cat([doc1_tokens, doc2_tokens]).unsqueeze(0)
    doc_lens = torch.tensor([doc1_len, doc2_len], device=device)

    batched_doc1 = doc1_tokens.unsqueeze(0)
    batched_doc2 = doc2_tokens.unsqueeze(0)

    print(f"Packed input shape: {packed_input.shape}")
    print(f"doc_lens: {doc_lens.tolist()}")
    print(f"Batched doc1 shape: {batched_doc1.shape}")
    print(f"Batched doc2 shape: {batched_doc2.shape}")

    with torch.no_grad():
        olmo_packed_logits = olmo_model(packed_input, doc_lens=doc_lens, max_doc_lens=[max(doc1_len, doc2_len)])
        olmo_batched_doc1_logits = olmo_model(batched_doc1)
        olmo_batched_doc2_logits = olmo_model(batched_doc2)

    olmo_packed_doc1 = olmo_packed_logits[0, :doc1_len]
    olmo_packed_doc2 = olmo_packed_logits[0, doc1_len:doc1_len + doc2_len]

    doc1_diff = (olmo_packed_doc1 - olmo_batched_doc1_logits[0]).abs()
    doc2_diff = (olmo_packed_doc2 - olmo_batched_doc2_logits[0]).abs()

    print(f"\nDoc1 (packed vs batched):")
    print(f"  Max diff: {doc1_diff.max().item():.6e}")
    print(f"  Mean diff: {doc1_diff.mean().item():.6e}")

    print(f"\nDoc2 (packed vs batched):")
    print(f"  Max diff: {doc2_diff.max().item():.6e}")
    print(f"  Mean diff: {doc2_diff.mean().item():.6e}")

    # Also compare HF batched vs OLMo batched
    with torch.no_grad():
        hf_doc1_logits = hf_model(batched_doc1).logits
        hf_doc2_logits = hf_model(batched_doc2).logits

    hf_olmo_doc1_diff = (hf_doc1_logits[0] - olmo_batched_doc1_logits[0]).abs()
    hf_olmo_doc2_diff = (hf_doc2_logits[0] - olmo_batched_doc2_logits[0]).abs()

    print(f"\nHF vs OLMo batched (doc1):")
    print(f"  Max diff: {hf_olmo_doc1_diff.max().item():.6e}")
    print(f"  Mean diff: {hf_olmo_doc1_diff.mean().item():.6e}")

    print(f"\nHF vs OLMo batched (doc2):")
    print(f"  Max diff: {hf_olmo_doc2_diff.max().item():.6e}")
    print(f"  Mean diff: {hf_olmo_doc2_diff.mean().item():.6e}")

    # Test 6: Verify reproducibility with fresh forward passes
    print("\n" + "=" * 60)
    print("TEST 6: REPRODUCIBILITY CHECK (fresh forward passes)")
    print("=" * 60)

    torch.manual_seed(42)
    test_input = torch.randint(1, 100352, (1, 15), device=device)

    with torch.no_grad():
        hf_pass1 = hf_model(test_input).logits
        hf_pass2 = hf_model(test_input).logits
        olmo_pass1 = olmo_model(test_input)
        olmo_pass2 = olmo_model(test_input)

    hf_self_diff = (hf_pass1 - hf_pass2).abs().max().item()
    olmo_self_diff = (olmo_pass1 - olmo_pass2).abs().max().item()
    hf_olmo_diff = (hf_pass1 - olmo_pass1).abs().max().item()

    print(f"HF self-consistency: {hf_self_diff:.6e}")
    print(f"OLMo self-consistency: {olmo_self_diff:.6e}")
    print(f"HF vs OLMo: {hf_olmo_diff:.6e}")

    # Test 7: Check if issue is with padding position encoding
    print("\n" + "=" * 60)
    print("TEST 7: SAME CONTENT, DIFFERENT PADDING LENGTHS")
    print("=" * 60)

    torch.manual_seed(42)
    content_tokens = torch.randint(1, 100352, (1, 10), device=device)

    input_no_pad = content_tokens
    input_pad_5 = torch.cat([content_tokens, torch.zeros(1, 5, dtype=torch.long, device=device)], dim=1)
    input_pad_10 = torch.cat([content_tokens, torch.zeros(1, 10, dtype=torch.long, device=device)], dim=1)

    attn_no_pad = torch.ones(1, 10, device=device)
    attn_pad_5 = torch.cat([torch.ones(1, 10, device=device), torch.zeros(1, 5, device=device)], dim=1)
    attn_pad_10 = torch.cat([torch.ones(1, 10, device=device), torch.zeros(1, 10, device=device)], dim=1)

    with torch.no_grad():
        hf_no_pad = hf_model(input_no_pad, attention_mask=attn_no_pad).logits
        hf_pad_5 = hf_model(input_pad_5, attention_mask=attn_pad_5).logits
        hf_pad_10 = hf_model(input_pad_10, attention_mask=attn_pad_10).logits

        olmo_no_pad = olmo_model(input_no_pad)
        olmo_pad_5 = olmo_model(input_pad_5)
        olmo_pad_10 = olmo_model(input_pad_10)

    hf_content_no_pad = hf_no_pad[0, :10]
    hf_content_pad_5 = hf_pad_5[0, :10]
    hf_content_pad_10 = hf_pad_10[0, :10]

    olmo_content_no_pad = olmo_no_pad[0, :10]
    olmo_content_pad_5 = olmo_pad_5[0, :10]
    olmo_content_pad_10 = olmo_pad_10[0, :10]

    print("HF content comparison (with vs without padding):")
    print(f"  no_pad vs pad_5: {(hf_content_no_pad - hf_content_pad_5).abs().max().item():.6e}")
    print(f"  no_pad vs pad_10: {(hf_content_no_pad - hf_content_pad_10).abs().max().item():.6e}")

    print("\nOLMo content comparison (with vs without padding):")
    print(f"  no_pad vs pad_5: {(olmo_content_no_pad - olmo_content_pad_5).abs().max().item():.6e}")
    print(f"  no_pad vs pad_10: {(olmo_content_no_pad - olmo_content_pad_10).abs().max().item():.6e}")

    print("\nHF vs OLMo (content region only):")
    print(f"  no_pad: {(hf_content_no_pad - olmo_content_no_pad).abs().max().item():.6e}")
    print(f"  pad_5: {(hf_content_pad_5 - olmo_content_pad_5).abs().max().item():.6e}")
    print(f"  pad_10: {(hf_content_pad_10 - olmo_content_pad_10).abs().max().item():.6e}")

    # Test 8: Systematic sequence length test
    print("\n" + "=" * 60)
    print("TEST 8: SYSTEMATIC SEQUENCE LENGTH TEST")
    print("=" * 60)

    print("Testing HF vs OLMo at different sequence lengths:")
    failed_lengths = []
    for seq_len_test in [1, 2, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 32, 48, 64, 128, 256, 512]:
        torch.manual_seed(42)
        test_input = torch.randint(1, 100352, (1, seq_len_test), device=device)
        with torch.no_grad():
            hf_out = hf_model(test_input).logits
            olmo_out = olmo_model(test_input)
        diff = (hf_out - olmo_out).abs().max().item()
        status = "✓" if diff == 0 else "✗"
        print(f"  seq_len={seq_len_test:3d}: max_diff={diff:.6e} {status}")
        if diff > 0:
            failed_lengths.append(seq_len_test)

    if failed_lengths:
        print(f"\nFailed lengths: {failed_lengths}")

    # Test 9: Compare RoPE embeddings directly
    print("\n" + "=" * 60)
    print("TEST 9: DIRECT ROPE COMPARISON")
    print("=" * 60)

    for test_len in [8, 10, 17, 20, 32]:
        torch.manual_seed(42)
        test_input = torch.randint(1, 100352, (1, test_len), device=device)
        position_ids = torch.arange(test_len, device=device).unsqueeze(0)

        with torch.no_grad():
            hf_cos, hf_sin = hf_model.model.rotary_emb(test_input, position_ids)
            olmo_pos_sin, olmo_pos_cos = olmo_model.blocks["0"].attention.rope._get_rotary_embedding(
                test_len, device
            )

        hf_cos_flat = hf_cos.squeeze()
        hf_sin_flat = hf_sin.squeeze()
        olmo_cos_flat = olmo_pos_cos[:test_len]
        olmo_sin_flat = olmo_pos_sin[:test_len]

        cos_diff = (hf_cos_flat - olmo_cos_flat).abs().max().item()
        sin_diff = (hf_sin_flat - olmo_sin_flat).abs().max().item()

        status = "✓" if cos_diff == 0 and sin_diff == 0 else "✗"
        print(f"  seq_len={test_len:3d}: cos_diff={cos_diff:.6e}, sin_diff={sin_diff:.6e} {status}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Test 1 (Simple forward): Perfect match expected")
    print("Test 2 (Padded input): Check if attention mask affects OLMo-core")
    print("Test 3 (Batched forward): Check batch consistency")
    print("Test 4 (Batched with padding): Real DPO scenario")
    print("Test 5 (Packed vs batched): Check doc_lens implementation")
    print("Test 6 (Reproducibility): Check self-consistency of both models")
    print("Test 7 (Padding lengths): Check if padding affects content logits")
    print("Test 8 (Sequence lengths): Check which lengths produce differences")


if __name__ == "__main__":
    main()
