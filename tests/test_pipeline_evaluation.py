import unittest
from backend.pipeline import Pipeline
from backend.Algorithm import VanillaRAG, GraphRAG
from backend.Dataset import Dataset
from ragas.dataset_schema import EvaluationResult

class TestPipelineEvaluation(unittest.TestCase):

    def setUp(self):
        """Set up test environment."""
        self.db_path = ":memory:"  # Use in-memory database for testing
        self.vanilla_rag = VanillaRAG()
        self.graph_rag = GraphRAG()
        self.pipeline_vanilla = Pipeline(db_path=self.db_path, algorithm=self.vanilla_rag)
        self.pipeline_graph = Pipeline(db_path=self.db_path, algorithm=self.graph_rag)

        # Create a mock dataset using the Dataset class
        self.mock_data = [
            {
                "question": "What is AI?",
                "contexts": ["AI stands for Artificial Intelligence.", 
                             "It is a branch of computer science."],
                "ground_truth": "Artificial Intelligence is a branch of computer science."
            },
            {
                "question": "What is machine learning?",
                "contexts": ["Machine learning is a subset of AI that focuses on algorithms."],
                "ground_truth": "Machine learning is a subset of AI that focuses on algorithms."
            }
        ]

    def test_vanilla_rag_evaluation(self):
        """Test evaluation with VanillaRAG."""
        # Add entries to the dataset
        for entry in self.mock_data:
            self.pipeline_vanilla.dataset.add_entry(
                question=entry["question"],
                contexts=entry["contexts"],
                response="",  # Initially empty
                ground_truth=entry["ground_truth"]
            )
        
        # Process the dataset
        self.pipeline_vanilla.process_query(self.pipeline_vanilla.dataset)
        
        # Evaluate the results
        results = self.pipeline_vanilla.evaluate()
        
        # Print the results
        print("\nVanillaRAG Evaluation Results:")
        print(results)
        
        # Check if evaluation results contain expected metrics
        self.assertIsInstance(results, EvaluationResult)

    def test_graph_rag_evaluation(self):
        """Test evaluation with GraphRAG."""
        # Add entries to the dataset
        for entry in self.mock_data:
            self.pipeline_graph.dataset.add_entry(
                question=entry["question"],
                contexts=entry["contexts"],
                response="",  # Initially empty
                ground_truth=entry["ground_truth"]
            )
        
        # Process the dataset
        self.pipeline_graph.process_query(self.pipeline_graph.dataset)
        
        # Evaluate the results
        results = self.pipeline_graph.evaluate()
        
        # Print the results
        print("\nGraphRAG Evaluation Results:")
        print(results)
        
        # Check if evaluation results contain expected metrics
        self.assertIsInstance(results, EvaluationResult)

    def tearDown(self):
        """Clean up after tests."""
        self.pipeline_vanilla.close()
        self.pipeline_graph.close()

if __name__ == '__main__':
    unittest.main() 