import unittest
from backend.pipeline import Pipeline
from backend.Algorithm import VanillaRAG
from ragas.dataset_schema import EvaluationResult

class TestSquadEvaluation(unittest.TestCase):

    def setUp(self):
        """Set up test environment."""
        self.db_path = ":memory:"  # Use in-memory database for testing
        self.vanilla_rag = VanillaRAG()
        self.pipeline = Pipeline(db_path=self.db_path, algorithm=self.vanilla_rag)

    def test_squad_dataset(self):
        """Test evaluation with the SQuAD dataset."""
        # Evaluate the SQuAD dataset using the method in the pipeline
        results = self.pipeline.evaluate_squad_dataset()

        # Print the results
        print("\nSQuAD Evaluation Results:")
        for result in results:
            print(f"Question: {result['question']}")
            print(f"Context: {result['context']}")
            print(f"Generated Response: {result['response']}")
            print(f"Ground Truth: {result['ground_truth']}")

        # # Check if evaluation results contain expected metrics
        # self.assertIsInstance(results, list)  # Ensure results are in list format
        # self.assertGreater(len(results), 0, "No results generated from the evaluation.")
        # Evaluate the results
        results = self.pipeline.evaluate_squad_dataset()
        
        # Print the results
        print("\nSQuAD Evaluation Results:")
        print(results)
        
        # Check if evaluation results contain expected metrics
        self.assertIsInstance(results, EvaluationResult)


    def tearDown(self):
        """Clean up after tests."""
        self.pipeline.close()

if __name__ == '__main__':
    unittest.main()