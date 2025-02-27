import os
import chromadb  # ✅ Correct import
from langchain.vectorstores import Chroma

from langchain.embeddings import HuggingFaceEmbeddings  # ✅ No API key required
from langchain.schema import Document

class VectorDB:
    def __init__(self, persist_directory="db_memory"):
        """
        Initialize ChromaDB vector storage with Hugging Face embeddings.
        """
        # ✅ Load Hugging Face sentence-transformer model
        self.embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # ✅ Ensure the persistence directory exists
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)

        # ✅ Initialize ChromaDB with Hugging Face Embeddings
        self.db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embed_model
        )

    def store_interaction(self, query, response):
        """
        Stores user queries and responses in ChromaDB for future recall.
        """
        doc = Document(page_content=query, metadata={"response": response})
        self.db.add_texts([query], metadatas=[{"response": response}])

    def retrieve_similar(self, query, k=2):
        """
        Retrieves past similar queries to provide context-aware responses.
        """
        results = self.db.similarity_search(query, k=k)
        return [doc.metadata["response"] for doc in results]

    def clear_memory(self):
        """
        Clears all stored interactions in the vector database.
        """
        self.db.delete_collection()
