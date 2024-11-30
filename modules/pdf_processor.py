import PyPDF2

def read_pdf(file_path):
    """
    Read and extract text from a PDF file.

    This function opens a PDF file, reads its content, and extracts text from each page.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF.
    """
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

def chunk_text(text, chunk_size=2000, chunk_overlap=500):
    """
    Split text into chunks using RecursiveCharacterTextSplitter.

    This function splits the input text into smaller chunks of a specified size with a specified overlap.

    Args:
        text (str): The text to be split into chunks.
        chunk_size (int, optional): The maximum size of each chunk. Defaults to 1000.
        chunk_overlap (int, optional): The number of characters to overlap between chunks. Defaults to 250.

    Returns:
        list[str]: A list of text chunks.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n", "\n\n"])
    return splitter.split_text(text)
