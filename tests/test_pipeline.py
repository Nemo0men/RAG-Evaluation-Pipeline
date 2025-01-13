import unittest
from backend.pipeline import Pipeline
from backend.Algorithm import VanillaRAG, GraphRAG

class TestPipeline(unittest.TestCase):

    def setUp(self):
        """Set up test environment."""
        self.db_path = ":memory:"  # Use in-memory database for testing
        self.vanilla_rag = VanillaRAG()
        self.graph_rag = GraphRAG()
        self.pipeline_vanilla = Pipeline(db_path=self.db_path, algorithm=self.vanilla_rag)
        self.pipeline_graph = Pipeline(db_path=self.db_path, algorithm=self.graph_rag)
        self.test_queries = [
            {
                "query_text": "What is AI?",
                "contexts": ["AI stands for Artificial Intelligence.", 
                             "It is a branch of computer science."],
                "ground_truth": "Artificial Intelligence is a branch of computer science."
            }
        ]

    def test_vanilla_rag_process_query(self):
        """Test processing a query with VanillaRAG."""
        result = self.pipeline_vanilla.process_query(
            self.test_queries[0]["query_text"],
            self.test_queries[0]["contexts"],
            self.test_queries[0]["ground_truth"]
        )
        self.assertIn("response", result)
        self.assertIsInstance(result["response"], str)

    def test_graph_rag_process_query(self):
        """Test processing a query with GraphRAG."""
        result = self.pipeline_graph.process_query(
            self.test_queries[0]["query_text"],
            self.test_queries[0]["contexts"],
            self.test_queries[0]["ground_truth"]
        )
        self.assertIn("response", result)
        self.assertIsInstance(result["response"], str)


    # these tests are not working
    def test_vanilla_rag_batch_process(self):
        """Test batch processing with VanillaRAG."""
        results = self.pipeline_vanilla.batch_process(self.test_queries)
        self.assertIsInstance(results, dict)
        self.assertIn("context_precision", results)

    def test_graph_rag_batch_process(self):
        """Test batch processing with GraphRAG."""
        results = self.pipeline_graph.batch_process(self.test_queries)
        self.assertIsInstance(results, dict)
        self.assertIn("context_precision", results)

    def tearDown(self):
        """Clean up after tests."""
        self.pipeline_vanilla.close()
        self.pipeline_graph.close()

if __name__ == '__main__':
    unittest.main() 