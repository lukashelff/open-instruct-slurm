"""GPU integration tests for DPO utils including TensorCache.

These tests require CUDA and will be skipped if not available.

To run:
    ./scripts/train/build_image_and_launch.sh scripts/test/run_gpu_pytest.sh
"""

import logging
import pathlib
import tempfile
import unittest

import torch
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.attention.backend import has_flash_attn_2
from olmo_core.nn.transformer import TransformerConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from open_instruct import dpo_utils, model_utils

logger = logging.getLogger(__name__)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestTensorCacheGPU(unittest.TestCase):
    def test_tensor_cache_gpu_indexing(self):
        cache = model_utils.TensorCache(
            tensors={
                "chosen_logps": torch.tensor([1.0, 2.0, 3.0, 4.0]).cuda(),
                "rejected_logps": torch.tensor([5.0, 6.0, 7.0, 8.0]).cuda(),
            }
        )
        indices = torch.tensor([0, 2]).cuda()
        result = cache[indices]

        self.assertEqual(result["chosen_logps"].device.type, "cuda")
        self.assertTrue(torch.equal(result["chosen_logps"], torch.tensor([1.0, 3.0]).cuda()))
        self.assertTrue(torch.equal(result["rejected_logps"], torch.tensor([5.0, 7.0]).cuda()))

    def test_tensor_cache_disk_roundtrip_with_gpu(self):
        cache = model_utils.TensorCache(tensors={"chosen_logps": torch.tensor([1.0, 2.0, 3.0]).cuda()})
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = pathlib.Path(tmpdir) / "cache.pt"
            cache.to_disk(cache_path)
            loaded_cache = model_utils.TensorCache.from_disk(cache_path, torch.device("cuda"))
            self.assertEqual(loaded_cache.tensors["chosen_logps"].device.type, "cuda")
            self.assertTrue(torch.allclose(cache.tensors["chosen_logps"], loaded_cache.tensors["chosen_logps"]))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestDataCollatorDatasetIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.model = AutoModelForCausalLM.from_pretrained(cls.model_name, torch_dtype=torch.bfloat16).cuda()

    def test_collator_preserves_index(self):
        samples = [
            {
                "chosen_input_ids": torch.tensor([1, 2, 3, 4, 5]),
                "chosen_labels": [-100, -100, 3, 4, 5],
                "chosen_attention_mask": torch.tensor([1, 1, 1, 1, 1]),
                "rejected_input_ids": torch.tensor([1, 2, 6, 7, 8]),
                "rejected_labels": [-100, -100, 6, 7, 8],
                "rejected_attention_mask": torch.tensor([1, 1, 1, 1, 1]),
                "index": i,
            }
            for i in range(4)
        ]

        collator = dpo_utils.DataCollatorForSeq2SeqDPO(tokenizer=self.tokenizer, model=self.model, padding="longest")
        batch = collator(samples)

        self.assertIn("index", batch)
        self.assertTrue(torch.equal(batch["index"], torch.tensor([0, 1, 2, 3])))


class OlmoStyleModel(torch.nn.Module):
    """Mock OLMo-style model that returns logits directly (not wrapped in an output object)."""

    def __init__(self, vocab_size: int = 1000):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, 64)
        self.linear = torch.nn.Linear(64, vocab_size)

    def forward(
        self, input_ids: torch.Tensor, doc_lens: torch.Tensor | None = None, max_doc_lens: list | None = None
    ) -> torch.Tensor:
        del doc_lens, max_doc_lens
        return self.linear(self.embed(input_ids))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestForwardFunctionsOlmo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.olmo_model = OlmoStyleModel().cuda().to(torch.bfloat16)
        cls.hf_model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
        cls.hf_model = AutoModelForCausalLM.from_pretrained(cls.hf_model_name, torch_dtype=torch.bfloat16).cuda()

    def _make_batch(self, batch_size: int = 2, seq_len: int = 10, vocab_size: int = 1000):
        return {
            "chosen_input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)).cuda(),
            "chosen_labels": torch.randint(0, vocab_size, (batch_size, seq_len)).cuda(),
            "chosen_attention_mask": torch.ones(batch_size, seq_len).cuda(),
            "rejected_input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)).cuda(),
            "rejected_labels": torch.randint(0, vocab_size, (batch_size, seq_len)).cuda(),
            "rejected_attention_mask": torch.ones(batch_size, seq_len).cuda(),
        }

    def test_concatenated_forward_olmo(self):
        batch = self._make_batch()
        chosen_logps, rejected_logps, aux_loss = dpo_utils.concatenated_forward_olmo(self.olmo_model, batch)

        self.assertEqual(chosen_logps.shape, (2,))
        self.assertEqual(rejected_logps.shape, (2,))
        self.assertIsNone(aux_loss)
        self.assertTrue(torch.isfinite(chosen_logps).all())
        self.assertTrue(torch.isfinite(rejected_logps).all())

    def test_separate_forward_olmo(self):
        batch = self._make_batch()
        chosen_logps, rejected_logps, aux_loss = dpo_utils.separate_forward_olmo(self.olmo_model, batch)

        self.assertEqual(chosen_logps.shape, (2,))
        self.assertEqual(rejected_logps.shape, (2,))
        self.assertIsNone(aux_loss)
        self.assertTrue(torch.isfinite(chosen_logps).all())
        self.assertTrue(torch.isfinite(rejected_logps).all())

    def test_concatenated_forward_hf(self):
        tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
        batch = self._make_batch(vocab_size=tokenizer.vocab_size)
        chosen_logps, rejected_logps, aux_loss = dpo_utils.concatenated_forward(self.hf_model, batch)

        self.assertEqual(chosen_logps.shape, (2,))
        self.assertEqual(rejected_logps.shape, (2,))
        self.assertIsNone(aux_loss)
        self.assertTrue(torch.isfinite(chosen_logps).all())
        self.assertTrue(torch.isfinite(rejected_logps).all())

    def test_separate_forward_hf(self):
        tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
        batch = self._make_batch(vocab_size=tokenizer.vocab_size)
        chosen_logps, rejected_logps, aux_loss = dpo_utils.separate_forward(self.hf_model, batch)

        self.assertEqual(chosen_logps.shape, (2,))
        self.assertEqual(rejected_logps.shape, (2,))
        self.assertIsNone(aux_loss)
        self.assertTrue(torch.isfinite(chosen_logps).all())
        self.assertTrue(torch.isfinite(rejected_logps).all())

    def test_olmo_and_hf_produce_different_results(self):
        batch = self._make_batch()
        olmo_chosen, olmo_rejected, _ = dpo_utils.concatenated_forward_olmo(self.olmo_model, batch)

        tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
        hf_batch = self._make_batch(vocab_size=tokenizer.vocab_size)
        hf_chosen, hf_rejected, _ = dpo_utils.concatenated_forward(self.hf_model, hf_batch)

        self.assertFalse(torch.allclose(olmo_chosen, hf_chosen))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
@unittest.skipUnless(has_flash_attn_2(), "Flash attention required for document masking")
class TestConcatenatedVsSeparateForwardOlmo(unittest.TestCase):
    """Test that concatenated_forward_olmo produces same results as separate_forward_olmo.

    This test verifies that packing chosen and rejected sequences together with doc_lens
    produces identical logits to processing them separately. If this fails, it indicates
    a bug in how document boundaries are handled in the packed attention.
    """

    @classmethod
    def setUpClass(cls):
        config = TransformerConfig.olmo2_1B(vocab_size=100352)
        config.n_layers = 2
        config.block.attention = AttentionConfig(
            n_heads=config.block.attention.n_heads,
            n_kv_heads=config.block.attention.n_kv_heads,
            bias=config.block.attention.bias,
            rope=config.block.attention.rope,
            qk_norm=config.block.attention.qk_norm,
            backend="flash_2",
        )
        cls.model = config.build().cuda().to(torch.bfloat16)

    def _make_batch_with_shared_prefix(self, prefix_len: int = 50, response_len: int = 20):
        """Create a batch where chosen and rejected share a prefix but have different responses."""
        vocab_size = 100352
        shared_prefix = torch.randint(1, vocab_size, (1, prefix_len)).cuda()
        chosen_response = torch.randint(1, vocab_size, (1, response_len)).cuda()
        rejected_response = torch.randint(1, vocab_size, (1, response_len)).cuda()

        chosen_ids = torch.cat([shared_prefix, chosen_response], dim=1)
        rejected_ids = torch.cat([shared_prefix, rejected_response], dim=1)

        chosen_labels = torch.cat([torch.full((1, prefix_len), -100, device="cuda"), chosen_response], dim=1)
        rejected_labels = torch.cat([torch.full((1, prefix_len), -100, device="cuda"), rejected_response], dim=1)

        return {
            "chosen_input_ids": chosen_ids,
            "chosen_labels": chosen_labels,
            "chosen_attention_mask": torch.ones_like(chosen_ids),
            "rejected_input_ids": rejected_ids,
            "rejected_labels": rejected_labels,
            "rejected_attention_mask": torch.ones_like(rejected_ids),
        }

    def test_concatenated_equals_separate_forward(self):
        """Verify concatenated and separate forward produce identical results."""
        torch.manual_seed(42)
        batch = self._make_batch_with_shared_prefix()

        with torch.no_grad():
            concat_chosen, concat_rejected, _ = dpo_utils.concatenated_forward_olmo(self.model, batch)
            sep_chosen, sep_rejected, _ = dpo_utils.separate_forward_olmo(self.model, batch)

        self.assertTrue(
            torch.allclose(concat_chosen, sep_chosen, atol=1e-4),
            f"Chosen logps differ: concat={concat_chosen.tolist()}, sep={sep_chosen.tolist()}",
        )
        self.assertTrue(
            torch.allclose(concat_rejected, sep_rejected, atol=1e-4),
            f"Rejected logps differ: concat={concat_rejected.tolist()}, sep={sep_rejected.tolist()}",
        )

    def test_raw_logits_packed_vs_separate(self):
        """Test raw OLMo-core model logits: packed with doc_lens vs separate forward passes."""
        seq_len = 10
        seq = torch.randint(1, 100352, (1, seq_len)).cuda()

        packed = torch.cat([seq, seq], dim=1)
        doc_lens = torch.tensor([seq_len, seq_len], device="cuda")

        with torch.no_grad():
            logits_packed = self.model(packed, doc_lens=doc_lens, max_doc_lens=[seq_len])
            logits_separate = self.model(seq)

        pos0_packed_doc1 = logits_packed[0, 0, :5].tolist()
        pos0_packed_doc2 = logits_packed[0, seq_len, :5].tolist()
        pos0_separate = logits_separate[0, 0, :5].tolist()

        pos1_packed_doc1 = logits_packed[0, 1, :5].tolist()
        pos1_packed_doc2 = logits_packed[0, seq_len + 1, :5].tolist()
        pos1_separate = logits_separate[0, 1, :5].tolist()

        logger.info(f"pos0 packed doc1: {pos0_packed_doc1}")
        logger.info(f"pos0 packed doc2: {pos0_packed_doc2}")
        logger.info(f"pos0 separate:    {pos0_separate}")
        logger.info(f"pos1 packed doc1: {pos1_packed_doc1}")
        logger.info(f"pos1 packed doc2: {pos1_packed_doc2}")
        logger.info(f"pos1 separate:    {pos1_separate}")

        pos0_doc1_matches_sep = torch.allclose(logits_packed[0, 0, :], logits_separate[0, 0, :], atol=1e-3, rtol=1e-3)
        pos0_doc2_matches_sep = torch.allclose(
            logits_packed[0, seq_len, :], logits_separate[0, 0, :], atol=1e-3, rtol=1e-3
        )
        pos1_doc1_matches_sep = torch.allclose(logits_packed[0, 1, :], logits_separate[0, 1, :], atol=1e-3, rtol=1e-3)
        pos1_doc2_matches_sep = torch.allclose(
            logits_packed[0, seq_len + 1, :], logits_separate[0, 1, :], atol=1e-3, rtol=1e-3
        )

        logger.info(f"pos0 doc1 matches separate: {pos0_doc1_matches_sep}")
        logger.info(f"pos0 doc2 matches separate: {pos0_doc2_matches_sep}")
        logger.info(f"pos1 doc1 matches separate: {pos1_doc1_matches_sep}")
        logger.info(f"pos1 doc2 matches separate: {pos1_doc2_matches_sep}")

        self.assertTrue(pos0_doc1_matches_sep, "pos0 doc1 should match separate")
        self.assertTrue(pos0_doc2_matches_sep, "pos0 doc2 should match separate (doc_lens should reset RoPE)")
        self.assertTrue(pos1_doc1_matches_sep, "pos1 doc1 should match separate")
        self.assertTrue(pos1_doc2_matches_sep, "pos1 doc2 should match separate (doc_lens should reset RoPE)")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
@unittest.skipUnless(has_flash_attn_2(), "Flash attention required for document masking")
class TestPackedVsBatchedForward(unittest.TestCase):
    """Test that packed forward with doc_lens produces DIFFERENT results than batched forward.

    This demonstrates the root cause of numerical differences between OLMo-core and HuggingFace
    DPO implementations: OLMo-core uses packed sequences with doc_lens, while HuggingFace uses
    standard batched forward. Even with identical inputs, these produce different logits.
    """

    @classmethod
    def setUpClass(cls):
        config = TransformerConfig.olmo2_1B(vocab_size=100352)
        config.n_layers = 2
        config.block.attention = AttentionConfig(
            n_heads=config.block.attention.n_heads,
            n_kv_heads=config.block.attention.n_kv_heads,
            bias=config.block.attention.bias,
            rope=config.block.attention.rope,
            qk_norm=config.block.attention.qk_norm,
            backend="flash_2",
        )
        cls.model = config.build().cuda().to(torch.bfloat16)

    def test_batched_vs_packed_identical_sequences(self):
        """Show that batched and packed forward give different results for identical sequences.

        When we have two identical sequences:
        - Batched forward: model(input_ids) with shape [2, seq_len]
        - Packed forward: model(packed_ids, doc_lens=...) with shape [1, 2*seq_len]

        Both should theoretically give the same results, but Flash Attention with doc_lens
        introduces numerical differences.
        """
        seq_len = 20
        seq = torch.randint(1, 100352, (1, seq_len)).cuda()

        batched_input = seq.repeat(2, 1)
        packed_input = seq.repeat(1, 2)
        doc_lens = torch.tensor([seq_len, seq_len], device="cuda")

        with torch.no_grad():
            logits_batched = self.model(batched_input)
            logits_packed = self.model(packed_input, doc_lens=doc_lens, max_doc_lens=[seq_len])

        logits_batched_seq0 = logits_batched[0]
        logits_batched_seq1 = logits_batched[1]
        logits_packed_doc0 = logits_packed[0, :seq_len]
        logits_packed_doc1 = logits_packed[0, seq_len:]

        batched_seq0_matches_seq1 = torch.allclose(logits_batched_seq0, logits_batched_seq1, atol=1e-5)
        packed_doc0_matches_doc1 = torch.allclose(logits_packed_doc0, logits_packed_doc1, atol=1e-5)
        batched_matches_packed_doc0 = torch.allclose(logits_batched_seq0, logits_packed_doc0, atol=1e-3)

        logger.info(f"Batched seq0 matches seq1 (should be True): {batched_seq0_matches_seq1}")
        logger.info(f"Packed doc0 matches doc1 (should be True): {packed_doc0_matches_doc1}")
        logger.info(f"Batched matches packed doc0 (may differ): {batched_matches_packed_doc0}")

        max_diff = (logits_batched_seq0 - logits_packed_doc0).abs().max().item()
        logger.info(f"Max absolute difference between batched and packed: {max_diff}")

        self.assertTrue(
            batched_seq0_matches_seq1, "Batched forward should give identical results for identical inputs"
        )
        self.assertTrue(packed_doc0_matches_doc1, "Packed forward should give identical results for identical docs")

    def test_dpo_forward_batched_vs_packed(self):
        """Compare DPO forward with batched (HF-style) vs packed (OLMo-style) approach.

        This test shows that the current concatenated_forward_olmo (which packs sequences)
        produces different logps than a hypothetical batched forward would.
        """
        prefix_len = 30
        response_len = 15
        vocab_size = 100352

        shared_prefix = torch.randint(1, vocab_size, (1, prefix_len)).cuda()
        chosen_response = torch.randint(1, vocab_size, (1, response_len)).cuda()
        rejected_response = torch.randint(1, vocab_size, (1, response_len)).cuda()

        chosen_ids = torch.cat([shared_prefix, chosen_response], dim=1)
        rejected_ids = torch.cat([shared_prefix, rejected_response], dim=1)

        chosen_labels = torch.cat([torch.full((1, prefix_len), -100, device="cuda"), chosen_response], dim=1)
        rejected_labels = torch.cat([torch.full((1, prefix_len), -100, device="cuda"), rejected_response], dim=1)

        batch = {
            "chosen_input_ids": chosen_ids,
            "chosen_labels": chosen_labels,
            "chosen_attention_mask": torch.ones_like(chosen_ids),
            "rejected_input_ids": rejected_ids,
            "rejected_labels": rejected_labels,
            "rejected_attention_mask": torch.ones_like(rejected_ids),
        }

        with torch.no_grad():
            packed_chosen, packed_rejected, _ = dpo_utils.concatenated_forward_olmo(self.model, batch)

            concatenated_batch = dpo_utils.concatenated_inputs(batch)
            input_ids = concatenated_batch["concatenated_input_ids"]
            labels = concatenated_batch["concatenated_labels"]
            batched_logits = self.model(input_ids)
            batched_all_logps = dpo_utils._get_batch_logps(batched_logits, labels)
            batched_chosen = batched_all_logps[:1]
            batched_rejected = batched_all_logps[1:]

        chosen_diff = (packed_chosen - batched_chosen).abs().item()
        rejected_diff = (packed_rejected - batched_rejected).abs().item()

        logger.info(f"Packed chosen logps: {packed_chosen.tolist()}")
        logger.info(f"Batched chosen logps: {batched_chosen.tolist()}")
        logger.info(f"Chosen logps difference: {chosen_diff}")

        logger.info(f"Packed rejected logps: {packed_rejected.tolist()}")
        logger.info(f"Batched rejected logps: {batched_rejected.tolist()}")
        logger.info(f"Rejected logps difference: {rejected_diff}")

        self.assertGreater(
            chosen_diff,
            0.01,
            f"Expected numerical difference between packed and batched forward, but got {chosen_diff}",
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestConcatenatedVsSeparateForwardHF(unittest.TestCase):
    """Test that concatenated_forward and separate_forward produce identical logps with HF models.

    concatenated_forward pads both sequences to max(chosen_len, rejected_len) and runs one
    forward pass with batch_size=2. separate_forward runs two forward passes, each at the
    sequence's own length. With correct attention masking, these should produce identical logps.
    """

    @classmethod
    def setUpClass(cls):
        cls.model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.model = AutoModelForCausalLM.from_pretrained(cls.model_name, torch_dtype=torch.bfloat16).cuda()

    def _make_batch_with_lengths(self, prefix_len: int, chosen_response_len: int, rejected_response_len: int):
        vocab_size = self.tokenizer.vocab_size
        shared_prefix = torch.randint(1, vocab_size, (1, prefix_len)).cuda()
        chosen_response = torch.randint(1, vocab_size, (1, chosen_response_len)).cuda()
        rejected_response = torch.randint(1, vocab_size, (1, rejected_response_len)).cuda()

        chosen_ids = torch.cat([shared_prefix, chosen_response], dim=1)
        rejected_ids = torch.cat([shared_prefix, rejected_response], dim=1)

        chosen_labels = torch.cat([torch.full((1, prefix_len), -100, device="cuda"), chosen_response], dim=1)
        rejected_labels = torch.cat([torch.full((1, prefix_len), -100, device="cuda"), rejected_response], dim=1)

        return {
            "chosen_input_ids": chosen_ids,
            "chosen_labels": chosen_labels,
            "chosen_attention_mask": torch.ones_like(chosen_ids),
            "rejected_input_ids": rejected_ids,
            "rejected_labels": rejected_labels,
            "rejected_attention_mask": torch.ones_like(rejected_ids),
        }

    def test_same_lengths(self):
        """Both sequences have the same length, so no padding difference."""
        torch.manual_seed(42)
        batch = self._make_batch_with_lengths(prefix_len=50, chosen_response_len=20, rejected_response_len=20)

        with torch.no_grad():
            concat_chosen, concat_rejected, _ = dpo_utils.concatenated_forward(self.model, batch)
            sep_chosen, sep_rejected, _ = dpo_utils.separate_forward(self.model, batch)

        self.assertTrue(
            torch.allclose(concat_chosen, sep_chosen, atol=1e-4),
            f"Chosen logps differ: concat={concat_chosen.tolist()}, sep={sep_chosen.tolist()}",
        )
        self.assertTrue(
            torch.allclose(concat_rejected, sep_rejected, atol=1e-4),
            f"Rejected logps differ: concat={concat_rejected.tolist()}, sep={sep_rejected.tolist()}",
        )

    def test_different_lengths(self):
        """Chosen is longer than rejected; rejected gets padded in concatenated_forward."""
        torch.manual_seed(42)
        batch = self._make_batch_with_lengths(prefix_len=50, chosen_response_len=70, rejected_response_len=30)

        with torch.no_grad():
            concat_chosen, concat_rejected, _ = dpo_utils.concatenated_forward(self.model, batch)
            sep_chosen, sep_rejected, _ = dpo_utils.separate_forward(self.model, batch)

        logger.info(f"concat_chosen={concat_chosen.tolist()}, sep_chosen={sep_chosen.tolist()}")
        logger.info(f"concat_rejected={concat_rejected.tolist()}, sep_rejected={sep_rejected.tolist()}")
        logger.info(f"chosen_diff={(concat_chosen - sep_chosen).abs().item()}")
        logger.info(f"rejected_diff={(concat_rejected - sep_rejected).abs().item()}")

        self.assertTrue(
            torch.allclose(concat_chosen, sep_chosen, atol=1e-4),
            f"Chosen logps differ: concat={concat_chosen.tolist()}, sep={sep_chosen.tolist()}",
        )
        self.assertTrue(
            torch.allclose(concat_rejected, sep_rejected, atol=1e-4),
            f"Rejected logps differ: concat={concat_rejected.tolist()}, sep={sep_rejected.tolist()}",
        )


if __name__ == "__main__":
    unittest.main()
