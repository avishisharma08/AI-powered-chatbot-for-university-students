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

from langchain_core.documents import Document
from rag_udemy import ask, REFUSAL_TEXT


class TestAskRefusalSources(unittest.TestCase):

    @patch("rag_udemy.app.db")
    @patch("rag_udemy.app.llm")
    def test_pure_refusal_clears_sources(self, mock_llm, mock_db):
        """When the answer is exactly equal to REFUSAL_TEXT, sources must be an empty list []."""
        mock_doc = Document(
            page_content="Anti-ragging rules and guidelines.",
            metadata={"source": "antiragging_policy.txt"}
        )
        # Mock passing similarity search (score <= 0.75 threshold)
        mock_db.similarity_search_with_score.return_value = [(mock_doc, 0.5)]

        mock_response = MagicMock()
        mock_response.content = REFUSAL_TEXT
        mock_llm.invoke.return_value = mock_response

        res = ask("What is the anti-ragging helpline number?")

        self.assertEqual(res["answer"].strip(), REFUSAL_TEXT)
        self.assertEqual(res["sources"], [])

    @patch("rag_udemy.app.db")
    @patch("rag_udemy.app.llm")
    def test_refusal_with_extra_text_keeps_sources(self, mock_llm, mock_db):
        """When the answer has REFUSAL_TEXT followed by extra text, sources must be kept."""
        mock_doc = Document(
            page_content="Uniform dress code details.",
            metadata={"source": "uniform_rules.txt"}
        )
        # Mock passing similarity search (score <= 0.75 threshold)
        mock_db.similarity_search_with_score.return_value = [(mock_doc, 0.5)]

        mock_response = MagicMock()
        mock_response.content = f"{REFUSAL_TEXT} However, students may contact the administrative office."
        mock_llm.invoke.return_value = mock_response

        res = ask("Is there a dress code for Sunday?")

        self.assertNotEqual(res["answer"].strip(), REFUSAL_TEXT)
        self.assertEqual(res["sources"], ["uniform_rules.txt"])


if __name__ == "__main__":
    unittest.main()
