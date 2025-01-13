from typing import List, Dict, Any
from .Dataset import Dataset
from .Metric import Metric
from .Algorithm import BaseRAGAlgorithm
from datasets import Dataset as HF_Dataset, load_dataset

class Pipeline:
    """
    Main pipeline class that orchestrates data processing, model inference, and evaluation.
    """
    def __init__(self, db_path: str, algorithm: BaseRAGAlgorithm):
        """
        Initialize the Pipeline with components.

        Parameters:
        - db_path: Path to the SQLite database for the Dataset
        - algorithm: An instance of a RAG algorithm class
        """
        self.dataset = Dataset(db_path=db_path)
        self.algorithm = algorithm
        self.metric = Metric()

    def process_query(self, dataset: Dataset) -> None:
        """Process a dataset through the pipeline."""
        for entry in dataset.get_all_entries().to_dict(orient='records'):
            response = self.algorithm.generate_response(
                entry["question"],
                entry["contexts"]  
            )
                # Update the existing entry with the response
            self.dataset.update_response(
                question=entry["question"],
                new_response=response  # Store the response in the response column
            )

    def evaluate_squad_dataset(self, sample_size: int = 10) -> HF_Dataset:
        """Evaluate the SQuAD dataset."""
        squad_dataset = load_dataset("rajpurkar/squad_v2", split='validation')
        small_subset = squad_dataset.select(range(sample_size))
        evaluation_data = []
        
        for entry in small_subset:
            evaluation_data.append({
                "question": entry["question"],
                "context": entry["context"],
                "ground_truth": entry["answers"]["text"]  # This is a list of possible answers
            })
        
        results = []
        for entry in evaluation_data:
            response = self.algorithm.generate_response(entry["question"], entry["context"])
            results.append({
                "question": entry["question"],
                "context": entry["context"],
                "response": response,
                "ground_truth": entry["ground_truth"]
            })
        
        # Create a Hugging Face dataset from the results
        hf_dataset = HF_Dataset.from_dict({
            "question": [item["question"] for item in results],
            "context": [item["context"] for item in results],
            "response": [item["response"] for item in results],
            "ground_truth": [item["ground_truth"] for item in results]
        })
        
        results = self.metric.evaluate(hf_dataset)
        
        return results
    
    

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate all entries in the dataset using configured metrics.
        """
        # Convert to HuggingFace dataset format
        eval_data = self.dataset.to_hf_dataset()
        
        # Run evaluation
        results = self.metric.evaluate(eval_data)
        
        return results

    def batch_process(self, queries: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Process multiple queries and evaluate results.

        Parameters:
        - queries: List of dicts containing query_text, contexts, and ground_truth
        """
        for query in queries:
            self.process_query(
                query["query_text"],
                query["contexts"],
                query["ground_truth"]
            )
        
        return self.evaluate()

    def close(self):
        """Close all resources."""
        self.dataset.close()


# Example Usage
if __name__ == "__main__":
    queries = [
        {
            "query_text": "What is AI?",
            "contexts": ["AI stands for Artificial Intelligence.", 
                        "It is a branch of computer science."],
            "ground_truth": "Artificial Intelligence is a branch of computer science."
        }
    ]

    # Initialize pipeline with VanillaRAG
    from .Algorithm import VanillaRAG
    vanilla_rag = VanillaRAG()
    pipeline = Pipeline(db_path="dataset.db", algorithm=vanilla_rag)

    try:
        # Process queries and get evaluation results
        results = pipeline.batch_process(queries)

        # Print results
        print("\nEvaluation Results:")
        print(results)

    finally:
        # Clean up
        pipeline.close()
