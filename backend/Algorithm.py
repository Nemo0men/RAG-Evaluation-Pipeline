from langchain_openai import ChatOpenAI, OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
from langchain.chains import GraphQAChain
from langchain.text_splitter import CharacterTextSplitter

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}

---
If there is no relevant information in the context, please say "Unable to find matching results."
"""

class BaseRAGAlgorithm:
    def __init__(self, llm):
        self.llm = llm

    def generate_response(self, query_text, context):
        raise NotImplementedError("This method should be implemented by subclasses.")

class VanillaRAG(BaseRAGAlgorithm):
    def __init__(self):
        self.model = ChatOpenAI()

    def generate_response(self, query_text, context, max_length=512, num_return_sequences=1):
        """
        Generate a response for a given query and context using the pipeline.

        Args:
            query_text (str): The input query.
            context (str): The context for the query.
            max_length (int): Maximum length of the response (default: 512).
            num_return_sequences (int): Number of response sequences to generate (default: 1).

        Returns:
            str: The generated response text.
        """
        # Construct the prompt using the context and query
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context, question=query_text.strip())

        # Generate the response using the language model
        response_text = self.model.invoke(prompt).content
        return response_text

class GraphRAG(VanillaRAG):
    def __init__(self):
        super().__init__()

    def generate_response(self, query_text, context, max_length=512, num_return_sequences=1):
        """
        Generate a response for a given query and context using the pipeline.

        Args:
            query_text (str): The input query.
            context (str): The context for the query.
            max_length (int): Maximum length of the response (default: 512).
            num_return_sequences (int): Number of response sequences to generate (default: 1).

        Returns:
            str: The generated response text.
        """
        # Use VanillaRAG's method to get the initial response
        initial_response = super().generate_response(query_text, context, max_length, num_return_sequences)

        # Check if the initial response indicates no results
        if initial_response == "Unable to find matching results.":
            return initial_response

        # Process with GraphRAG
        documents = [Document(page_content=context)]
        llm = OpenAI()
        llm_transformer = LLMGraphTransformer(llm=llm)
        text_splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=20, separator='\n')
        texts = text_splitter.split_documents(documents)
        graph_documents = llm_transformer.convert_to_graph_documents(texts)

        graph = NetworkxEntityGraph()

        for doc in graph_documents:
            for node in doc.nodes:
                graph.add_node(node.id)
            for edge in doc.relationships:
                graph._graph.add_edge(
                    edge.source.id,
                    edge.target.id,
                    relation=edge.type,
                )

        chain = GraphQAChain.from_llm(llm=llm, graph=graph, verbose=True)
        refined_context = chain.invoke({"query": query_text.strip()})

        # Combine refined context with original contexts
        combined_context = f"{refined_context}\n\n---\n\n{context}"

        # Generate Answer
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=combined_context, question=query_text.strip())

        response_text = self.model.invoke(prompt).content
        return response_text
