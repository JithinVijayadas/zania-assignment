from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from concurrent.futures import ThreadPoolExecutor
from config import Config
from .prompts import SYSTEM_ANSWER_PROMPT, HUMAN_ANSWER_PROMPT
from .confidence_calculator import calculate_confidence_score

def answer_question(question, vector_store, model, user_instruction, chunks):
    """
    Generate an answer for a given question using the provided model and vector store.

    This function performs a similarity search on the vector store to find relevant documents,
    constructs messages for the model, and invokes the model to generate an answer. It also
    calculates the confidence score of the answer and adjusts the answer if the confidence is low.

    Args:
        question (str): The question to be answered.
        vector_store: The vector store containing the document chunks.
        model: The language model to be used for generating the answer.
        user_instruction (str): Additional instructions for the model.
        chunks (list[str]): The document chunks.

    Returns:
        tuple: A tuple containing the question and the generated answer.
    """
    docs = vector_store.similarity_search(question, k=1)
    messages = [
        SystemMessage(content=SYSTEM_ANSWER_PROMPT + user_instruction),
        HumanMessage(content=HUMAN_ANSWER_PROMPT.format(input_document=chunks, question=question))
    ]
    answer = model.invoke(messages).content
    if "Data Not Available" in answer:
        confidence = calculate_confidence_score(answer, docs, model)
    else:
        confidence = 100.00
    if confidence < 90.00:
        answer = "Data Not Available"
    return question, answer

def process_pdf(file_path, questions, user_instruction):
    """
    Process a PDF file and generate answers for a list of questions.

    This function reads the PDF file, splits the text into chunks, creates a FAISS vector store,
    and uses a language model to generate answers for the provided questions.

    Args:
        file_path (str): The path to the PDF file.
        questions (list[str]): A list of questions to be answered.
        user_instruction (str): Additional instructions for the model.

    Returns:
        dict: A dictionary mapping each question to its generated answer.
    """
    from modules.pdf_processor import read_pdf, chunk_text
    from modules.vector_store import create_faiss_store

    text = read_pdf(file_path)
    chunks = chunk_text(text)
    vector_store = create_faiss_store(chunks)
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=Config.OPENAI_API_KEY)

    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda q: answer_question(q, vector_store, llm, user_instruction, chunks), questions)
    
    return dict(results)