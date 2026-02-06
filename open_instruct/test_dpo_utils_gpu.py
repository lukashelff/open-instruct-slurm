"""GPU integration tests for DPO utils including TensorCache.

These tests require CUDA and will be skipped if not available.

To run:
    ./scripts/train/build_image_and_launch.sh scripts/test/run_gpu_pytest.sh
"""

import pathlib
import tempfile
import unittest

import torch
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.attention.backend import has_flash_attn_2
from olmo_core.nn.hf import convert as olmo_hf_convert
from olmo_core.nn.transformer import TransformerConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from open_instruct import dpo_utils, hf_matched_olmo, logger_utils, model_utils, olmo_core_utils

logger = logger_utils.setup_logger(__name__)


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
        cls.model.train()
        for module in cls.model.modules():
            if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d, torch.nn.AlphaDropout)):
                module.p = 0.0

    @staticmethod
    def _snapshot_rng_state() -> dict[str, object]:
        state: dict[str, object] = {"cpu": torch.get_rng_state()}
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state: dict[str, object]) -> None:
        torch.set_rng_state(state["cpu"])  # type: ignore[arg-type]
        if "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])  # type: ignore[arg-type]

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
            rng_state = self._snapshot_rng_state()
            concat_chosen, concat_rejected, _ = dpo_utils.concatenated_forward(self.model, batch)
            self._restore_rng_state(rng_state)
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
            rng_state = self._snapshot_rng_state()
            concat_chosen, concat_rejected, _ = dpo_utils.concatenated_forward(self.model, batch)
            self._restore_rng_state(rng_state)
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


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
@unittest.skipUnless(has_flash_attn_2(), "Flash attention required")
class TestOlmoCoreVsHFGradientDivergence(unittest.TestCase):
    """Verify hypothesis 1: OLMo-core and HF model implementations produce different gradients.

    Even with identical weights and inputs, the different model implementations
    (different attention code paths, different autograd graphs) produce different
    gradients, causing training divergence after the first optimizer step.
    """

    @classmethod
    def setUpClass(cls):
        hf_config = AutoConfig.from_pretrained("allenai/OLMo-2-0425-1B")
        hf_config.num_hidden_layers = 2
        hf_config._attn_implementation = "flash_attention_2"
        cls.hf_model = AutoModelForCausalLM.from_config(hf_config).to(torch.bfloat16).cuda()
        cls.vocab_size = hf_config.vocab_size

        olmo_config = olmo_core_utils.get_transformer_config(
            "allenai/OLMo-2-0425-1B", hf_config.vocab_size, attn_backend="flash_2"
        )
        olmo_config.n_layers = 2
        olmo_core_utils.patch_rope_for_hf_compatibility()
        cls.olmo_model = olmo_config.build(init_device="cpu").to(torch.bfloat16).cuda()

        hf_state = cls.hf_model.state_dict()
        converted = olmo_hf_convert.convert_state_from_hf(
            hf_config, hf_state, model_type=getattr(hf_config, "model_type", None)
        )
        converted_gpu = {k: v.to(device="cuda") for k, v in converted.items()}
        cls.olmo_model.load_state_dict(converted_gpu, assign=True, strict=False)

        hf_params = sum(p.numel() for p in cls.hf_model.parameters())
        olmo_params = sum(p.numel() for p in cls.olmo_model.parameters())
        logger.info(f"HF model params: {hf_params}, OLMo-core params: {olmo_params}")

    def _make_batch(self):
        prefix_len = 50
        response_len = 20
        shared_prefix = torch.randint(1, self.vocab_size, (1, prefix_len)).cuda()
        chosen_response = torch.randint(1, self.vocab_size, (1, response_len)).cuda()
        rejected_response = torch.randint(1, self.vocab_size, (1, response_len)).cuda()

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

    def test_forward_close_but_gradients_differ(self):
        torch.manual_seed(42)
        batch = self._make_batch()

        self.hf_model.zero_grad()
        hf_chosen, hf_rejected, _ = dpo_utils.separate_forward(self.hf_model, batch)
        hf_loss = -(hf_chosen - hf_rejected).sigmoid().log().mean()
        hf_loss.backward()
        hf_grad_norm = (
            sum(p.grad.float().norm().item() ** 2 for p in self.hf_model.parameters() if p.grad is not None) ** 0.5
        )

        self.olmo_model.zero_grad()
        olmo_chosen, olmo_rejected, _ = dpo_utils.separate_forward_olmo(self.olmo_model, batch)
        olmo_loss = -(olmo_chosen - olmo_rejected).sigmoid().log().mean()
        olmo_loss.backward()
        olmo_grad_norm = (
            sum(p.grad.float().norm().item() ** 2 for p in self.olmo_model.parameters() if p.grad is not None) ** 0.5
        )

        logps_diff = (hf_chosen - olmo_chosen).abs().max().item()
        loss_diff = (hf_loss - olmo_loss).abs().item()
        grad_norm_diff = abs(hf_grad_norm - olmo_grad_norm)
        grad_norm_rel = grad_norm_diff / max(hf_grad_norm, olmo_grad_norm)

        logger.info(f"Logps max diff: {logps_diff}")
        logger.info(f"Loss diff: {loss_diff}")
        logger.info(f"HF grad norm: {hf_grad_norm:.6f}")
        logger.info(f"OLMo grad norm: {olmo_grad_norm:.6f}")
        logger.info(f"Grad norm abs diff: {grad_norm_diff:.6f}")
        logger.info(f"Grad norm rel diff: {grad_norm_rel:.6f}")

        self.assertNotAlmostEqual(
            hf_grad_norm,
            olmo_grad_norm,
            places=4,
            msg="Expected gradient norms to differ between OLMo-core and HF models",
        )

    def test_divergence_increases_after_optimizer_step(self):
        torch.manual_seed(42)
        batch = self._make_batch()

        hf_state = {k: v.clone() for k, v in self.hf_model.state_dict().items()}
        olmo_state = {k: v.clone() for k, v in self.olmo_model.state_dict().items()}

        self.hf_model.zero_grad()
        self.olmo_model.zero_grad()

        with torch.no_grad():
            hf_chosen0, _, _ = dpo_utils.separate_forward(self.hf_model, batch)
            olmo_chosen0, _, _ = dpo_utils.separate_forward_olmo(self.olmo_model, batch)
        initial_diff = (hf_chosen0 - olmo_chosen0).abs().item()

        hf_opt = torch.optim.SGD(self.hf_model.parameters(), lr=1e-3)
        olmo_opt = torch.optim.SGD(self.olmo_model.parameters(), lr=1e-3)

        self.hf_model.zero_grad()
        hf_chosen, hf_rejected, _ = dpo_utils.separate_forward(self.hf_model, batch)
        (-(hf_chosen - hf_rejected).sigmoid().log().mean()).backward()
        hf_opt.step()

        self.olmo_model.zero_grad()
        olmo_chosen, olmo_rejected, _ = dpo_utils.separate_forward_olmo(self.olmo_model, batch)
        (-(olmo_chosen - olmo_rejected).sigmoid().log().mean()).backward()
        olmo_opt.step()

        with torch.no_grad():
            hf_chosen1, _, _ = dpo_utils.separate_forward(self.hf_model, batch)
            olmo_chosen1, _, _ = dpo_utils.separate_forward_olmo(self.olmo_model, batch)
        post_step_diff = (hf_chosen1 - olmo_chosen1).abs().item()

        logger.info(f"Initial logps diff: {initial_diff:.6f}")
        logger.info(f"Post-step logps diff: {post_step_diff:.6f}")
        logger.info(f"Divergence ratio: {post_step_diff / max(initial_diff, 1e-10):.2f}x")

        self.hf_model.load_state_dict(hf_state)
        self.olmo_model.load_state_dict(olmo_state)

        self.assertGreater(
            post_step_diff,
            initial_diff,
            f"Expected logps to diverge more after optimizer step "
            f"(initial={initial_diff:.6f}, post_step={post_step_diff:.6f})",
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
@unittest.skipUnless(has_flash_attn_2(), "Flash attention required")
class TestRoPEPatchEffect(unittest.TestCase):
    """Test whether the RoPE CPU patch affects gradient divergence between OLMo-core and HF."""

    def _build_olmo_model_without_patch(self, hf_config, vocab_size):
        """Build OLMo-core model WITHOUT the RoPE CPU patch."""
        import olmo_core.nn.rope as rope_module  # noqa: PLC0415

        original_fn = rope_module.compute_inv_freqs
        rope_module.compute_inv_freqs = olmo_core_utils._original_compute_inv_freqs

        olmo_config = olmo_core_utils.get_transformer_config(
            "allenai/OLMo-2-0425-1B", vocab_size, attn_backend="flash_2"
        )
        olmo_config.n_layers = 2
        model = olmo_config.build(init_device="cpu").to(torch.bfloat16).cuda()

        rope_module.compute_inv_freqs = original_fn
        return model

    def _build_olmo_model_with_patch(self, hf_config, vocab_size):
        """Build OLMo-core model WITH the RoPE CPU patch."""
        olmo_core_utils.patch_rope_for_hf_compatibility()

        olmo_config = olmo_core_utils.get_transformer_config(
            "allenai/OLMo-2-0425-1B", vocab_size, attn_backend="flash_2"
        )
        olmo_config.n_layers = 2
        return olmo_config.build(init_device="cpu").to(torch.bfloat16).cuda()

    def _make_batch(self, vocab_size):
        prefix_len = 50
        response_len = 20
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

    def _compute_grad_norm(self, model):
        return sum(p.grad.float().norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5

    def test_rope_patch_reduces_divergence(self):
        """Test if RoPE CPU patch reduces gradient divergence between OLMo-core and HF."""
        hf_config = AutoConfig.from_pretrained("allenai/OLMo-2-0425-1B")
        hf_config.num_hidden_layers = 2
        hf_config._attn_implementation = "flash_attention_2"
        hf_model = AutoModelForCausalLM.from_config(hf_config).to(torch.bfloat16).cuda()
        vocab_size = hf_config.vocab_size

        unpatched_olmo = self._build_olmo_model_without_patch(hf_config, vocab_size)
        patched_olmo = self._build_olmo_model_with_patch(hf_config, vocab_size)

        hf_state = hf_model.state_dict()
        converted = olmo_hf_convert.convert_state_from_hf(
            hf_config, hf_state, model_type=getattr(hf_config, "model_type", None)
        )
        converted_gpu = {k: v.to(device="cuda") for k, v in converted.items()}

        unpatched_olmo.load_state_dict(converted_gpu, assign=True, strict=False)
        patched_olmo.load_state_dict({k: v.clone() for k, v in converted_gpu.items()}, assign=True, strict=False)

        torch.manual_seed(42)
        batch = self._make_batch(vocab_size)

        hf_model.zero_grad()
        hf_chosen, hf_rejected, _ = dpo_utils.separate_forward(hf_model, batch)
        hf_loss = -(hf_chosen - hf_rejected).sigmoid().log().mean()
        hf_loss.backward()
        hf_grad_norm = self._compute_grad_norm(hf_model)

        unpatched_olmo.zero_grad()
        unpatched_chosen, unpatched_rejected, _ = dpo_utils.separate_forward_olmo(unpatched_olmo, batch)
        unpatched_loss = -(unpatched_chosen - unpatched_rejected).sigmoid().log().mean()
        unpatched_loss.backward()
        unpatched_grad_norm = self._compute_grad_norm(unpatched_olmo)

        patched_olmo.zero_grad()
        patched_chosen, patched_rejected, _ = dpo_utils.separate_forward_olmo(patched_olmo, batch)
        patched_loss = -(patched_chosen - patched_rejected).sigmoid().log().mean()
        patched_loss.backward()
        patched_grad_norm = self._compute_grad_norm(patched_olmo)

        unpatched_grad_diff = abs(hf_grad_norm - unpatched_grad_norm)
        patched_grad_diff = abs(hf_grad_norm - patched_grad_norm)

        unpatched_logps_diff = (hf_chosen - unpatched_chosen).abs().item()
        patched_logps_diff = (hf_chosen - patched_chosen).abs().item()

        print("\n=== RoPE Patch Effect Results ===")
        print(f"HF grad norm: {hf_grad_norm:.6f}")
        print(f"Unpatched OLMo grad norm: {unpatched_grad_norm:.6f} (diff from HF: {unpatched_grad_diff:.6f})")
        print(f"Patched OLMo grad norm: {patched_grad_norm:.6f} (diff from HF: {patched_grad_diff:.6f})")
        print(f"Unpatched logps diff from HF: {unpatched_logps_diff:.6f}")
        print(f"Patched logps diff from HF: {patched_logps_diff:.6f}")

        if patched_grad_diff < unpatched_grad_diff:
            improvement = (unpatched_grad_diff - patched_grad_diff) / unpatched_grad_diff * 100
            print(f"RESULT: RoPE patch REDUCES gradient divergence by {improvement:.1f}%")
        else:
            print("RESULT: RoPE patch does NOT reduce gradient divergence")
        print("=================================\n")

        self.assertGreater(
            unpatched_grad_diff + patched_grad_diff,
            0.0,
            f"Expected gradient divergence. HF={hf_grad_norm:.6f}, "
            f"Unpatched={unpatched_grad_norm:.6f} (diff={unpatched_grad_diff:.6f}), "
            f"Patched={patched_grad_norm:.6f} (diff={patched_grad_diff:.6f})",
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
@unittest.skipUnless(has_flash_attn_2(), "Flash attention required")
class TestAttentionLayerEquivalence(unittest.TestCase):
    """Pinpoint whether the attention layer, norm, or MLP causes OLMo-core vs HF divergence.

    Compares layer-by-layer activations and per-module gradients between OLMo-core
    and HF models loaded with identical weights, using the same input.
    """

    @classmethod
    def setUpClass(cls):
        hf_config = AutoConfig.from_pretrained("allenai/OLMo-2-0425-1B")
        hf_config.num_hidden_layers = 2
        hf_config._attn_implementation = "flash_attention_2"
        cls.hf_model = AutoModelForCausalLM.from_config(hf_config).to(torch.bfloat16).cuda()
        cls.vocab_size = hf_config.vocab_size

        olmo_config = olmo_core_utils.get_transformer_config(
            "allenai/OLMo-2-0425-1B", hf_config.vocab_size, attn_backend="flash_2"
        )
        olmo_config.n_layers = 2
        olmo_core_utils.patch_rope_for_hf_compatibility()
        cls.olmo_model = olmo_config.build(init_device="cpu").to(torch.bfloat16).cuda()

        hf_state = cls.hf_model.state_dict()
        converted = olmo_hf_convert.convert_state_from_hf(
            hf_config, hf_state, model_type=getattr(hf_config, "model_type", None)
        )
        converted_gpu = {k: v.to(device="cuda") for k, v in converted.items()}
        cls.olmo_model.load_state_dict(converted_gpu, assign=True, strict=False)

    @staticmethod
    def _make_output_hook(storage, key):
        def hook(module, args, output):
            if isinstance(output, tuple):
                storage[key] = output[0].detach().clone()
            else:
                storage[key] = output.detach().clone()

        return hook

    @staticmethod
    def _diff(a, b):
        af = a.float()
        bf = b.float()
        return (af - bf).abs().max().item(), (af - bf).abs().mean().item()

    def test_layer_by_layer_forward_comparison(self):
        """Compare activations at each sub-module to find where divergence starts."""
        torch.manual_seed(42)
        input_ids = torch.randint(1, self.vocab_size, (1, 70)).cuda()

        hf_acts = {}
        olmo_acts = {}
        hooks = []

        hooks.append(self.hf_model.model.embed_tokens.register_forward_hook(self._make_output_hook(hf_acts, "embed")))
        for i in range(2):
            layer = self.hf_model.model.layers[i]
            hooks.append(layer.self_attn.register_forward_hook(self._make_output_hook(hf_acts, f"L{i}_attn")))
            hooks.append(
                layer.post_attention_layernorm.register_forward_hook(
                    self._make_output_hook(hf_acts, f"L{i}_attn_norm")
                )
            )
            hooks.append(layer.mlp.register_forward_hook(self._make_output_hook(hf_acts, f"L{i}_mlp")))
            hooks.append(
                layer.post_feedforward_layernorm.register_forward_hook(
                    self._make_output_hook(hf_acts, f"L{i}_ff_norm")
                )
            )
            hooks.append(layer.register_forward_hook(self._make_output_hook(hf_acts, f"L{i}_block")))

        hooks.append(self.olmo_model.embeddings.register_forward_hook(self._make_output_hook(olmo_acts, "embed")))
        for i in range(2):
            block = self.olmo_model.blocks[str(i)]
            hooks.append(block.attention.register_forward_hook(self._make_output_hook(olmo_acts, f"L{i}_attn")))
            hooks.append(
                block.attention_norm.register_forward_hook(self._make_output_hook(olmo_acts, f"L{i}_attn_norm"))
            )
            hooks.append(block.feed_forward.register_forward_hook(self._make_output_hook(olmo_acts, f"L{i}_mlp")))
            hooks.append(
                block.feed_forward_norm.register_forward_hook(self._make_output_hook(olmo_acts, f"L{i}_ff_norm"))
            )
            hooks.append(block.register_forward_hook(self._make_output_hook(olmo_acts, f"L{i}_block")))

        with torch.no_grad():
            hf_logits = self.hf_model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits
            olmo_logits = self.olmo_model(input_ids)

        for h in hooks:
            h.remove()

        keys = ["embed"]
        for i in range(2):
            keys.extend([f"L{i}_attn", f"L{i}_attn_norm", f"L{i}_mlp", f"L{i}_ff_norm", f"L{i}_block"])

        logger.info("=== Layer-by-Layer Activation Comparison ===")
        first_divergence = None
        for key in keys:
            hf_val = hf_acts.get(key)
            olmo_val = olmo_acts.get(key)
            if hf_val is None or olmo_val is None:
                logger.info(f"{key}: MISSING (hf={hf_val is not None}, olmo={olmo_val is not None})")
                continue
            max_d, mean_d = self._diff(hf_val, olmo_val)
            logger.info(f"{key}: max_diff={max_d:.10f}, mean_diff={mean_d:.10f}")
            if first_divergence is None and max_d > 0:
                first_divergence = key

        logit_max, logit_mean = self._diff(hf_logits, olmo_logits)
        logger.info(f"logits: max_diff={logit_max:.10f}, mean_diff={logit_mean:.10f}")
        logger.info(f"First divergence at: {first_divergence}")

        embed_max, _ = self._diff(hf_acts["embed"], olmo_acts["embed"])
        self.assertEqual(embed_max, 0.0, "Embeddings should be bit-for-bit identical")

        self.assertIsNotNone(first_divergence, "Expected some divergence between models")

    def test_per_module_gradient_comparison(self):
        """Compare gradient norms per module to find which component has largest gradient diff."""
        torch.manual_seed(42)
        input_ids = torch.randint(1, self.vocab_size, (1, 70)).cuda()
        labels = input_ids.clone()
        labels[:, :35] = -100

        self.hf_model.zero_grad()
        hf_out = self.hf_model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
        hf_logps = dpo_utils._get_batch_logps(hf_out.logits, labels)
        hf_logps.sum().backward()

        self.olmo_model.zero_grad()
        olmo_logits = self.olmo_model(input_ids)
        olmo_logps = dpo_utils._get_batch_logps(olmo_logits, labels)
        olmo_logps.sum().backward()

        hf_module_grads = {}
        for i in range(2):
            layer = self.hf_model.model.layers[i]
            for name, submod in [
                ("attn_q", layer.self_attn.q_proj),
                ("attn_k", layer.self_attn.k_proj),
                ("attn_v", layer.self_attn.v_proj),
                ("attn_o", layer.self_attn.o_proj),
                ("attn_qnorm", layer.self_attn.q_norm),
                ("attn_knorm", layer.self_attn.k_norm),
                ("post_attn_norm", layer.post_attention_layernorm),
                ("mlp_gate", layer.mlp.gate_proj),
                ("mlp_up", layer.mlp.up_proj),
                ("mlp_down", layer.mlp.down_proj),
                ("post_ff_norm", layer.post_feedforward_layernorm),
            ]:
                grad_norm = (
                    sum(p.grad.float().norm().item() ** 2 for p in submod.parameters() if p.grad is not None) ** 0.5
                )
                hf_module_grads[f"L{i}_{name}"] = grad_norm

        olmo_module_grads = {}
        for i in range(2):
            block = self.olmo_model.blocks[str(i)]
            for name, submod in [
                ("attn_q", block.attention.w_q),
                ("attn_k", block.attention.w_k),
                ("attn_v", block.attention.w_v),
                ("attn_o", block.attention.w_out),
                ("attn_qnorm", block.attention.q_norm),
                ("attn_knorm", block.attention.k_norm),
                ("post_attn_norm", block.attention_norm),
                ("mlp_gate", block.feed_forward.w1),
                ("mlp_up", block.feed_forward.w3),
                ("mlp_down", block.feed_forward.w2),
                ("post_ff_norm", block.feed_forward_norm),
            ]:
                grad_norm = (
                    sum(p.grad.float().norm().item() ** 2 for p in submod.parameters() if p.grad is not None) ** 0.5
                )
                olmo_module_grads[f"L{i}_{name}"] = grad_norm

        logger.info("=== Per-Module Gradient Norm Comparison ===")
        logger.info(f"{'Module':<25} {'HF':>12} {'OLMo':>12} {'Abs Diff':>12} {'Rel Diff':>12}")
        logger.info("-" * 75)
        max_rel_diff_key = None
        max_rel_diff_val = 0.0
        for key in sorted(hf_module_grads.keys()):
            hf_g = hf_module_grads[key]
            olmo_g = olmo_module_grads.get(key, 0.0)
            abs_diff = abs(hf_g - olmo_g)
            rel_diff = abs_diff / max(hf_g, olmo_g, 1e-10)
            logger.info(f"{key:<25} {hf_g:>12.6f} {olmo_g:>12.6f} {abs_diff:>12.6f} {rel_diff:>12.6f}")
            if rel_diff > max_rel_diff_val:
                max_rel_diff_val = rel_diff
                max_rel_diff_key = key

        logger.info(f"Largest relative gradient diff: {max_rel_diff_key} ({max_rel_diff_val:.6f})")

        total_hf = sum(hf_module_grads.values())
        total_olmo = sum(olmo_module_grads.values())
        self.assertNotAlmostEqual(
            total_hf, total_olmo, places=2, msg="Expected gradient norms to differ between OLMo-core and HF"
        )

    def test_hf_matched_model_exact_match(self):
        hf_config = AutoConfig.from_pretrained("allenai/OLMo-2-0425-1B")
        hf_config.num_hidden_layers = 2
        hf_config._attn_implementation = "flash_attention_2"

        matched_model = hf_matched_olmo.HFMatchedOlmo2.from_hf_config(hf_config).to(torch.bfloat16).cuda()
        hf_state = self.hf_model.state_dict()
        converted = hf_matched_olmo.HFMatchedOlmo2.convert_hf_state_dict(hf_state)
        result = matched_model.load_state_dict(converted, strict=False)
        self.assertEqual(len(result.missing_keys), 0, f"Missing keys: {result.missing_keys}")

        torch.manual_seed(42)
        input_ids = torch.randint(1, self.vocab_size, (1, 70)).cuda()

        with torch.no_grad():
            hf_logits = self.hf_model(input_ids=input_ids, use_cache=False).logits
            matched_logits = matched_model(input_ids)

        max_diff = (hf_logits.float() - matched_logits.float()).abs().max().item()
        logger.info(f"HF vs HFMatched logits max_diff: {max_diff}")
        self.assertEqual(max_diff, 0.0, f"Expected bit-for-bit identical logits, got max_diff={max_diff}")

        labels = input_ids.clone()
        labels[:, :35] = -100

        self.hf_model.zero_grad()
        hf_out = self.hf_model(input_ids=input_ids, use_cache=False)
        hf_logps = dpo_utils._get_batch_logps(hf_out.logits, labels)
        hf_logps.sum().backward()
        hf_grad_norm = (
            sum(p.grad.float().norm().item() ** 2 for p in self.hf_model.parameters() if p.grad is not None) ** 0.5
        )

        matched_model.zero_grad()
        matched_logits_train = matched_model(input_ids)
        matched_logps = dpo_utils._get_batch_logps(matched_logits_train, labels)
        matched_logps.sum().backward()
        matched_grad_norm = (
            sum(p.grad.float().norm().item() ** 2 for p in matched_model.parameters() if p.grad is not None) ** 0.5
        )

        grad_diff = abs(hf_grad_norm - matched_grad_norm)
        logger.info(
            f"HF grad norm: {hf_grad_norm:.6f}, HFMatched grad norm: {matched_grad_norm:.6f}, diff: {grad_diff:.6f}"
        )
        self.assertAlmostEqual(
            hf_grad_norm,
            matched_grad_norm,
            places=4,
            msg=f"Gradient norms should match: HF={hf_grad_norm}, Matched={matched_grad_norm}",
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
@unittest.skipUnless(has_flash_attn_2(), "Flash attention required")
class TestFlashAttnVarlenVsStandardGradients(unittest.TestCase):
    """Disproved hypothesis 2: flash_attn_func and flash_attn_varlen_func produce identical gradients.

    Originally we hypothesized that different CUDA kernels would produce different backward
    gradients. Testing showed the gradients are exactly identical (diff=0.0), so the flash
    attention API difference is NOT a source of divergence between OLMo-core and HF.
    """

    def test_single_sequence_gradients_identical(self):
        import flash_attn  # noqa: PLC0415

        torch.manual_seed(42)
        seq_len, nheads, headdim = 64, 16, 128

        q_data = torch.randn(1, seq_len, nheads, headdim, device="cuda", dtype=torch.bfloat16)
        k_data = torch.randn(1, seq_len, nheads, headdim, device="cuda", dtype=torch.bfloat16)
        v_data = torch.randn(1, seq_len, nheads, headdim, device="cuda", dtype=torch.bfloat16)

        q1 = q_data.clone().requires_grad_(True)
        k1 = k_data.clone().requires_grad_(True)
        v1 = v_data.clone().requires_grad_(True)
        out1 = flash_attn.flash_attn_func(q1, k1, v1, causal=True)
        out1.sum().backward()

        q2 = q_data.squeeze(0).clone().requires_grad_(True)
        k2 = k_data.squeeze(0).clone().requires_grad_(True)
        v2 = v_data.squeeze(0).clone().requires_grad_(True)
        cu = torch.tensor([0, seq_len], device="cuda", dtype=torch.int32)
        out2 = flash_attn.flash_attn_varlen_func(q2, k2, v2, cu, cu, seq_len, seq_len, causal=True)
        out2.sum().backward()

        forward_diff = (out1.squeeze(0) - out2).abs().max().item()
        q_grad_diff = (q1.grad.squeeze(0) - q2.grad).abs().max().item()
        k_grad_diff = (k1.grad.squeeze(0) - k2.grad).abs().max().item()
        v_grad_diff = (v1.grad.squeeze(0) - v2.grad).abs().max().item()
        max_grad_diff = max(q_grad_diff, k_grad_diff, v_grad_diff)

        logger.info(f"Single-seq forward max diff: {forward_diff}")
        logger.info(f"Single-seq max gradient diff: {max_grad_diff}")

        self.assertEqual(forward_diff, 0.0, "Forward outputs should be exactly identical")
        self.assertEqual(max_grad_diff, 0.0, "Gradients should be exactly identical")

    def test_multiple_sequences_gradients_identical(self):
        import flash_attn  # noqa: PLC0415

        torch.manual_seed(42)
        batch_size, seq_len, nheads, headdim = 2, 64, 16, 128

        q_data = torch.randn(batch_size, seq_len, nheads, headdim, device="cuda", dtype=torch.bfloat16)
        k_data = torch.randn(batch_size, seq_len, nheads, headdim, device="cuda", dtype=torch.bfloat16)
        v_data = torch.randn(batch_size, seq_len, nheads, headdim, device="cuda", dtype=torch.bfloat16)

        q1 = q_data.clone().requires_grad_(True)
        k1 = k_data.clone().requires_grad_(True)
        v1 = v_data.clone().requires_grad_(True)
        out1 = flash_attn.flash_attn_func(q1, k1, v1, causal=True)
        out1.sum().backward()

        q2 = q_data.reshape(-1, nheads, headdim).clone().requires_grad_(True)
        k2 = k_data.reshape(-1, nheads, headdim).clone().requires_grad_(True)
        v2 = v_data.reshape(-1, nheads, headdim).clone().requires_grad_(True)
        cu = torch.tensor([0, seq_len, 2 * seq_len], device="cuda", dtype=torch.int32)
        out2 = flash_attn.flash_attn_varlen_func(q2, k2, v2, cu, cu, seq_len, seq_len, causal=True)
        out2.sum().backward()

        forward_diff = (out1.reshape(-1, nheads, headdim) - out2).abs().max().item()
        q_grad_diff = (q1.grad.reshape(-1, nheads, headdim) - q2.grad).abs().max().item()
        k_grad_diff = (k1.grad.reshape(-1, nheads, headdim) - k2.grad).abs().max().item()
        v_grad_diff = (v1.grad.reshape(-1, nheads, headdim) - v2.grad).abs().max().item()
        max_grad_diff = max(q_grad_diff, k_grad_diff, v_grad_diff)

        logger.info(f"Multi-seq forward max diff: {forward_diff}")
        logger.info(f"Multi-seq max gradient diff: {max_grad_diff}")

        self.assertEqual(forward_diff, 0.0, "Forward outputs should be exactly identical")
        self.assertEqual(max_grad_diff, 0.0, "Gradients should be exactly identical")


if __name__ == "__main__":
    unittest.main()
