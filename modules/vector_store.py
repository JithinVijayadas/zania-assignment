from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from config import Config

def create_faiss_store(chunks):
    """
    Create a FAISS vector store from text chunks.

    This function initializes OpenAI embeddings using the provided API key,
    and creates a FAISS vector store from the given text chunks. The FAISS
    vector store is used for efficient similarity search.

    Args:
        chunks (list[str]): A list of text chunks to be stored in the FAISS vector store.

    Returns:
        FAISS: An instance of the FAISS vector store containing the text chunks.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
    return FAISS.from_texts(chunks, embeddings)