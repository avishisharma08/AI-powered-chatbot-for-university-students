import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure src is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from rag_udemy import config
from rag_udemy.verifier import verify, parse_verifier_json


class TestVerifier(unittest.TestCase):

    @patch("rag_udemy.verifier.get_llm")
    def test_1_fully_supported_example(self, mock_get_llm):
        """Test 1: Hand-written example where the statement is fully supported by context."""
        mock_response = MagicMock()
        mock_response.content = """
        [
            {
                "sentence": "Students must maintain at least 75% attendance in each subject to appear in semester exams.",
                "label": "SUPPORTED",
                "reason": "Directly supported by the policy text requiring 75% minimum attendance."
            }
        ]
        """
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        chunks = [
            "Attendance Policy: Students must maintain at least 75% attendance in each subject to be eligible to appear in semester examinations."
        ]
        answer = "Students must maintain at least 75% attendance in each subject to appear in semester exams."

        results = verify(answer, chunks)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["label"], "SUPPORTED")
        self.assertEqual(results[0]["sentence"], answer)
        self.assertIn("75%", results[0]["reason"])

    @patch("rag_udemy.verifier.get_llm")
    def test_2_unsupported_hallucination_example(self, mock_get_llm):
        """Test 2: Hand-written example where the statement is an unsupported hallucination."""
        mock_response = MagicMock()
        mock_response.content = """
        [
            {
                "sentence": "The library opens at 6:00 AM on Sundays and holidays.",
                "label": "UNSUPPORTED",
                "reason": "Context mentions library hours are notified by institution, but does not state 6:00 AM for Sundays."
            }
        ]
        """
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        chunks = [
            "The library operates during working hours notified by the institution. Check the latest notice for opening and closing hours."
        ]
        answer = "The library opens at 6:00 AM on Sundays and holidays."

        results = verify(answer, chunks)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["label"], "UNSUPPORTED")
        self.assertIn("Sunday", results[0]["reason"])

    @patch("rag_udemy.verifier.get_llm")
    def test_3_mixed_sentences_example(self, mock_get_llm):
        """Test 3: Hand-written example with one supported and one unsupported sentence."""
        mock_response = MagicMock()
        mock_response.content = """
        [
            {
                "sentence": "A late fine of ₹5 per book per day may be charged for overdue books.",
                "label": "SUPPORTED",
                "reason": "Directly stated in the library rules."
            },
            {
                "sentence": "The maximum overdue fine is capped at ₹50.",
                "label": "UNSUPPORTED",
                "reason": "Context states maximum limit must be confirmed from official policy, no ₹50 cap is stated."
            }
        ]
        """
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        chunks = [
            "A late fine of ₹5 per book per day may be charged for overdue books. The actual fine and maximum limit must be confirmed from official policy."
        ]
        answer = "A late fine of ₹5 per book per day may be charged for overdue books. The maximum overdue fine is capped at ₹50."

        results = verify(answer, chunks)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["label"], "SUPPORTED")
        self.assertEqual(results[1]["label"], "UNSUPPORTED")

    @patch("rag_udemy.verifier.get_llm")
    def test_4_parsing_error_handling(self, mock_get_llm):
        """Test 4: Graceful handling of invalid or malformed JSON from the LLM."""
        # Non-JSON plain text
        mock_response = MagicMock()
        mock_response.content = "I reviewed the text and everything looks accurate."
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        chunks = ["Anti-ragging policy strictly prohibits harassment."]
        answer = "Ragging is prohibited."

        results = verify(answer, chunks)
        self.assertEqual(results, [])

        # Direct test on malformed JSON string
        malformed = "[{'sentence': 'Broken JSON without double quotes', label: ...}]"
        parsed = parse_verifier_json(malformed)
        self.assertEqual(parsed, [])

    @patch("rag_udemy.verifier.get_llm")
    def test_5_use_verifier_flag_disabled(self, mock_get_llm):
        """Test 5: USE_VERIFIER flag switches off verification and skips LLM call."""
        original_flag = config.USE_VERIFIER
        try:
            config.USE_VERIFIER = False

            chunks = ["Some university rule."]
            answer = "Some university statement."

            results = verify(answer, chunks)

            self.assertEqual(results, [])
            mock_get_llm.assert_not_called()
        finally:
            config.USE_VERIFIER = original_flag


if __name__ == "__main__":
    unittest.main()
