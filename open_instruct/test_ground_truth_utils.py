#!/usr/bin/env python3
"""
Test script for verifier functionality in Python
"""

import unittest
from unittest.mock import patch

from parameterized import parameterized

from open_instruct.ground_truth_utils import F1Verifier, PuzzleMatcherVerifier, SLRBenchVerifier
from open_instruct.ground_truth_utils import SLRBenchVerifierConfig
from open_instruct.slr.slr_verifier import evaluate_prediction


class TestPuzzleMatcherVerifier(unittest.TestCase):
    """Test suite for PuzzleMatcherVerifier"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.verifier = PuzzleMatcherVerifier()

    @parameterized.expand(
        [
            ("simple_match", "The answer is 42", "answer is 42", 1.0),
            ("with_thinking_tags", "<think>Let me solve this</think>Paris", "paris", 1.0),
            ("with_answer_tags", "<answer>New York City!</answer>", "new york city", 1.0),
            ("should_fail", "Wrong answer", "correct answer", 0.0),
            (
                "complex_example",
                "<think>This is about geography</think><answer>The capital of France is Paris.</answer>",
                "capital of france is paris",
                1.0,
            ),
        ]
    )
    def test_basic_scenarios(self, name, prediction, label, expected_score):
        """Test basic puzzle matcher scenarios from quick_test"""
        result = self.verifier([], prediction, label)
        self.assertEqual(result.score, expected_score)

    @parameterized.expand(
        [
            # Basic matching tests
            ("exact_match_numbers", "42", "42", 1.0),
            ("exact_match_text", "hello world", "hello world", 1.0),
            ("case_insensitive", "Hello World", "hello world", 1.0),
            # Tests with thinking tags
            ("thinking_tags_match", "<think>Let me think about this...</think>42", "42", 1.0),
            ("thinking_tags_text_match", "<think>This is complex</think>hello world", "hello world", 1.0),
            ("thinking_tags_no_match", "<think>Analysis...</think>Wrong Answer", "42", 0.0),
            # Tests with answer tags
            ("answer_tags_match", "<answer>42</answer>", "42", 1.0),
            ("answer_tags_text_match", "<answer>hello world</answer>", "hello world", 1.0),
            ("answer_tags_no_match", "<answer>wrong</answer>", "42", 0.0),
            # Combined tags tests
            ("both_tags_match", "<think>Thinking...</think><answer>42</answer>", "42", 1.0),
            (
                "both_tags_text_match",
                "<think>Let me solve this step by step</think><answer>hello world</answer>",
                "hello world",
                1.0,
            ),
            # Punctuation and articles tests
            ("remove_articles_punctuation", "The answer is 42!", "answer is 42", 1.0),
            ("remove_article_a", "A simple test.", "simple test", 1.0),
            ("remove_punctuation", "Hello, world!", "hello world", 1.0),
            # Whitespace tests
            ("normalize_whitespace", "  hello   world  ", "hello world", 1.0),
            ("replace_tabs_newlines", "hello\tworld\n", "hello world", 1.0),
            # Non-matching tests
            ("numbers_no_match", "42", "43", 0.0),
            ("text_no_match", "hello", "world", 0.0),
            ("empty_vs_nonempty", "", "42", 0.0),
            # English examples
            ("capital_city", "<answer>London</answer>", "london", 1.0),
            ("animal_with_article", "<think>Animal question</think>The elephant", "elephant", 1.0),
            ("scientist_name", "<answer>Albert Einstein</answer>", "albert einstein", 1.0),
            ("literature_reference", "Romeo and Juliet by Shakespeare", "romeo and juliet by shakespeare", 1.0),
            ("country_name", "<answer>United States of America</answer>", "united states of america", 1.0),
        ]
    )
    def test_puzzle_matcher_scenarios(self, name, prediction, label, expected_score):
        """Test various puzzle matcher scenarios"""
        result = self.verifier([], prediction, label)
        self.assertEqual(
            result.score, expected_score, f"Failed for {name}: prediction='{prediction}', label='{label}'"
        )


class TestF1Verifier(unittest.TestCase):
    """Test suite for F1Verifier"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.verifier = F1Verifier()

    @parameterized.expand(
        [
            # Basic F1 tests with single string label
            ("exact_match", "hello world", "hello world", 1.0),
            ("partial_match", "hello world", "hello", 2 / 3),  # precision=0.5, recall=1.0, f1=2/3
            ("no_match", "hello world", "goodbye", 0.0),
            # Thinking section removal
            ("with_thinking", "<think>Let me think...</think>hello world", "hello world", 1.0),
            ("with_thinking_partial", "<think>Analysis</think>hello world", "hello", 2 / 3),
            # Answer tag removal
            ("with_answer_tags", "<answer>hello world</answer>", "hello world", 1.0),
            # Combined tags
            ("both_tags", "<think>Thinking...</think><answer>hello world</answer>", "hello world", 1.0),
        ]
    )
    def test_single_label(self, name, prediction, label, expected_score):
        """Test F1 verifier with single string label"""
        result = self.verifier([], prediction, label)
        self.assertAlmostEqual(
            result.score,
            expected_score,
            places=5,
            msg=f"Failed for {name}: prediction='{prediction}', label='{label}'",
        )

    @parameterized.expand(
        [
            # List of labels - should return max F1
            ("first_matches_best", "hello world", ["hello world", "goodbye"], 1.0),
            ("second_matches_best", "hello world", ["goodbye", "hello world"], 1.0),
            ("partial_matches", "hello world", ["hello", "world"], 2 / 3),  # both have same F1
            ("none_match_well", "hello world", ["foo", "bar", "baz"], 0.0),
            # Single element list should behave same as string
            ("single_element_list", "hello world", ["hello world"], 1.0),
            # With thinking section
            ("list_with_thinking", "<think>hmm</think>hello world", ["goodbye", "hello world"], 1.0),
        ]
    )
    def test_list_labels(self, name, prediction, labels, expected_score):
        """Test F1 verifier with list of labels (should return max)"""
        result = self.verifier([], prediction, labels)
        self.assertAlmostEqual(
            result.score,
            expected_score,
            places=5,
            msg=f"Failed for {name}: prediction='{prediction}', labels={labels}",
        )


class TestSLRBenchVerifier(unittest.TestCase):
    """Tests for SLRBenchVerifier helper logic and SLR runtime edge cases."""

    def test_extract_prolog_rule_tiered(self):
        tagged = "<think>reasoning</think><answer>[RULE] eastbound(T) :- has_car(T,C). [/RULE]</answer>"
        codeblock = "<answer>```prolog\neastbound(T) :- has_car(T,C).\n```</answer>"
        plain = "eastbound(T) :- has_car(T,C)."

        tagged_rule, tagged_format_ok = SLRBenchVerifier._extract_prolog_rule(tagged)
        code_rule, code_format_ok = SLRBenchVerifier._extract_prolog_rule(codeblock)
        plain_rule, plain_format_ok = SLRBenchVerifier._extract_prolog_rule(plain)

        self.assertEqual(tagged_rule, "eastbound(T) :- has_car(T,C).")
        self.assertTrue(tagged_format_ok)

        self.assertEqual(code_rule, "eastbound(T) :- has_car(T,C).")
        self.assertFalse(code_format_ok)

        self.assertIsNone(plain_rule)
        self.assertFalse(plain_format_ok)

    def test_compute_reward_sparse_and_syntax_agnostic(self):
        # No free floor reward for wrong predictions when gate is enabled.
        self.assertEqual(SLRBenchVerifier.compute_reward(0.0, 0.0, 1.0, 0.8, k=6, partial_gate=0.5), 0.0)
        self.assertEqual(SLRBenchVerifier.compute_reward(0.0, 0.49, 1.0, 0.8, k=6, partial_gate=0.5), 0.0)

        # syntax_score is intentionally ignored by current design.
        no_syntax = SLRBenchVerifier.compute_reward(0.0, 0.7, 0.0, 0.8, k=6, partial_gate=0.5)
        with_syntax = SLRBenchVerifier.compute_reward(0.0, 0.7, 1.0, 0.8, k=6, partial_gate=0.5)
        self.assertEqual(no_syntax, with_syntax)

        # Full accuracy should remain near 1.0.
        full = SLRBenchVerifier.compute_reward(1.0, 1.0, 1.0, 0.8)
        self.assertGreaterEqual(full, 0.95)
        self.assertLessEqual(full, 1.0)

    def test_evaluate_prediction_handles_missing_swipl(self):
        validation_program = "eastbound(train0).\nwestbound(train1)."
        eval_config = {"positive_predicate": "eastbound", "negative_predicate": "westbound"}

        with patch("open_instruct.slr.slr_verifier.subprocess.run", side_effect=FileNotFoundError("swipl missing")):
            result = evaluate_prediction(
                prediction="eastbound(T) :- true.",
                validation_program=validation_program,
                eval_config=eval_config,
                timeout=1,
                isomorphic=False,
            )

        self.assertFalse(result["is_correct"])
        self.assertEqual(result["partial_score"], 0.0)
        self.assertFalse(result["syntax_valid"])
        self.assertIn("swipl", result["error"])

    def test_selected_score_follows_slr_reward_config(self):
        label = {
            "validation_program": "eastbound(train0).\nwestbound(train1).",
            "evaluation_config": {"positive_predicate": "eastbound", "negative_predicate": "westbound"},
        }
        prediction = "<answer>[RULE] eastbound(T) :- true. [/RULE]</answer>"

        def fake_eval(_rule, _vp, _cfg, timeout=5, isomorphic=True):
            return {
                "is_correct": False,
                "partial_score": 0.9 if isomorphic else 0.6,
                "syntax_valid": True,
            }

        # base-selected run should return base score, while still tracking both.
        verifier_base = SLRBenchVerifier(SLRBenchVerifierConfig(slr_reward="base"))
        verifier_base._evaluate_prediction = fake_eval
        result_base = verifier_base([], prediction, label)
        self.assertEqual(result_base.score, result_base.extra_scores["slr_bench_base"])
        self.assertEqual(result_base.extra_scores["slr_bench_selected"], result_base.extra_scores["slr_bench_base"])
        self.assertEqual(result_base.extra_scores["slr_bench_selected_is_base"], 1.0)
        self.assertEqual(result_base.extra_scores["slr_bench_selected_is_isomorphic"], 0.0)

        # isomorphic-selected run should return isomorphic score.
        verifier_iso = SLRBenchVerifier(SLRBenchVerifierConfig(slr_reward="isomorphic"))
        verifier_iso._evaluate_prediction = fake_eval
        result_iso = verifier_iso([], prediction, label)
        self.assertEqual(result_iso.score, result_iso.extra_scores["slr_bench_isomorphic"])
        self.assertEqual(
            result_iso.extra_scores["slr_bench_selected"], result_iso.extra_scores["slr_bench_isomorphic"]
        )
        self.assertEqual(result_iso.extra_scores["slr_bench_selected_is_base"], 0.0)
        self.assertEqual(result_iso.extra_scores["slr_bench_selected_is_isomorphic"], 1.0)


if __name__ == "__main__":
    unittest.main()
