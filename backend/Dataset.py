from typing import List, Dict, Any
import sqlite3
import pandas as pd
from datasets import Dataset as HF_Dataset

class Dataset:
    """
    Manages data storage and retrieval for the pipeline.
    """
    def __init__(self, db_path: str = "dataset.db"):
        """Initialize database connection and structure."""
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path)
        self._initialize_table()

    def _initialize_table(self):
        """Create the database table if it doesn't exist."""
        query = """
        CREATE TABLE IF NOT EXISTS dataset (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            contexts TEXT,
            response TEXT,
            ground_truth TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.connection.execute(query)
        self.connection.commit()

    def add_entry(self, question: str, contexts: List[str], 
                  response: str, ground_truth: str):
        """Add a new entry to the dataset."""
        query = """
        INSERT INTO dataset (question, contexts, response, ground_truth)
        VALUES (?, ?, ?, ?)
        """
        self.connection.execute(
            query, 
            (question, "\n".join(contexts), response, ground_truth)
        )
        self.connection.commit()

    def update_response(self, question: str, new_response: str) -> None:
        """Update the response for a specific question."""
        query = """
        UPDATE dataset
        SET response = ?, timestamp = CURRENT_TIMESTAMP
        WHERE question = ?
        """
        self.connection.execute(
            query, 
            (new_response, question)
        )
        self.connection.commit()

    def to_hf_dataset(self) -> HF_Dataset:
        """Convert stored data to HuggingFace Dataset format."""
        df = self.get_all_entries()
        df['contexts'] = df['contexts'].apply(lambda x: x.split('\n'))
        return HF_Dataset.from_pandas(df)

    def get_all_entries(self) -> pd.DataFrame:
        """Retrieve all entries as a pandas DataFrame."""
        return pd.read_sql_query("SELECT * FROM dataset", self.connection)

    def close(self):
        """Close the database connection."""
        if hasattr(self, 'connection') and self.connection:
            self.connection.commit()  
            self.connection.close()
            self.connection = None  # Set to None to prevent further access

