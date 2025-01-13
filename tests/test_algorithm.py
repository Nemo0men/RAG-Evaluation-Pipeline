import unittest
from backend.Algorithm import VanillaRAG, GraphRAG

class TestAlgorithm(unittest.TestCase):

    def setUp(self):
        """Set up test environment."""
        self.vanilla_rag = VanillaRAG()
        self.graph_rag = GraphRAG()
        self.query_text = "What is AI?"
        self.context = "AI stands for Artificial Intelligence. It is a branch of computer science."

    def test_vanilla_rag_generate_response(self):
        """Test VanillaRAG's response generation."""
        response = self.vanilla_rag.generate_response(self.query_text, self.context)
        self.assertIsInstance(response, str)
        self.assertNotEqual(response, "Unable to find matching results.")

    def test_graph_rag_generate_response(self):
        """Test GraphRAG's response generation."""
        response = self.graph_rag.generate_response(self.query_text, self.context)
        self.assertIsInstance(response, str)
        self.assertNotEqual(response, "Unable to find matching results.")

    def test_vanilla_rag_no_results(self):
        """Test VanillaRAG's behavior with no matching results."""
        response = self.vanilla_rag.generate_response(self.query_text, "")
        self.assertEqual(response, "Unable to find matching results.")

    def test_graph_rag_no_results(self):
        """Test GraphRAG's behavior with no matching results."""
        response = self.graph_rag.generate_response(self.query_text, "")
        self.assertEqual(response, "Unable to find matching results.")

if __name__ == '__main__':
    unittest.main() 