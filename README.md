# GraphRAG vs. RAG: Advancing Complex Data Analysis for Government Datasets

## Project Overview

This project, conducted by ML@UVA for the Logistics Management Institute (LMI), explores Microsoft's GraphRAG technology and other RAG algorithms to enhance retrieval-augmented generation (RAG) for complex, multi-hop reasoning questions on government-specific datasets[1][2].

## Objectives

- Compare GraphRAG and traditional RAG approaches for government data analysis
- Evaluate the effectiveness of GraphRAG in handling complex queries and multi-hop reasoning
- Assess the potential improvements in data security and compliance for federal agencies

## Key Features

- Implementation of GraphRAG and traditional RAG algorithms
- Analysis of performance metrics for both approaches
- Focus on government-specific use cases and datasets
- Exploration of data security and compliance enhancements

## Technologies Used

- Microsoft's GraphRAG
- Traditional RAG algorithms
- Large Language Models (LLMs)
- Knowledge graph technologies

## Installation

```bash
git clone https://github.com/your-repo/graphrag-vs-rag.git
cd graphrag-vs-rag
pip install -r requirements.txt
```

## Usage

```python
from graphrag import GraphRAG
from traditional_rag import TraditionalRAG

# Example usage of GraphRAG
graph_rag = GraphRAG(dataset_path="path/to/government/dataset")
result = graph_rag.query("Complex multi-hop question")

# Example usage of Traditional RAG
trad_rag = TraditionalRAG(dataset_path="path/to/government/dataset")
result = trad_rag.query("Simple question")
```

## Results

Our findings are still being evaluated.

## Future Work

- Implement GraphRAG for specific federal agency use cases
- Explore integration with existing government data systems
- Conduct thorough security and compliance assessments

## Contributors

- ML@UVA Team
- Logistics Management Institute (LMI)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
