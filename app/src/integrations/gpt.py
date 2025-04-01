from typing import Optional
from g4f.client import Client
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from app.src.config import settings


class G4FLLM:
    def __init__(self):
        self._client = Client()

    def __call__(self, prompt: str, **kwargs) -> str:
        response = self._client.chat.completions.create(
            model=settings.gpt_model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content


class GPTClient:
    def __init__(self):
        self.model = settings.gpt_model
        self.vectorstore = None
        self.llm = G4FLLM()

    def load_documents(self, file_path: str) -> None:
        """Load and index documents from CSV file"""
        loader = CSVLoader(file_path=file_path)
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = FAISS.from_documents(texts, embeddings)

    def query_documents(self, question: str) -> str:
        """Query documents with a question"""
        if not self.vectorstore:
            raise ValueError("Documents not loaded. Call load_documents() first.")
        
        # Perform similarity search
        docs = self.vectorstore.similarity_search(question)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Generate response using LLM
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        return self.llm(prompt)