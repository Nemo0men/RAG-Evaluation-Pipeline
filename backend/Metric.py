from typing import Dict, Any, List
from datasets import Dataset as HF_Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_similarity,
    answer_correctness,
)

class Metric:
    def __init__(self, selected_metrics: List[Any] = None):
        """Initialize metrics."""
        # Define the default metrics to be used for evaluation
        default_metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_similarity,
            answer_correctness,
        ]
        
        # Use selected metrics if provided, otherwise use default metrics
        self.metrics = selected_metrics if selected_metrics is not None else default_metrics

    def evaluate(self, dataset: HF_Dataset) -> Dict[str, float]:
        """Evaluate the dataset using RAGAS metrics."""
        if not isinstance(dataset, HF_Dataset):
            raise ValueError("Dataset must be a HuggingFace Dataset")

        # Run evaluation using RAGAS
        results = evaluate(
            dataset=dataset,
            metrics=self.metrics
        )

        # # Convert results to a dictionary
        # results_dict = results.to_dict()
        
        # Return the results
        return results

