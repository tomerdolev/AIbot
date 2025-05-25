import os
import logging
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize
from langchain.document_loaders import PyPDFLoader,Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter 
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# Provide a standardized and professional way to output information, warnings, and errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure necessary NLTK resources are downloaded ( for sentence splitting)
# Check if punkt tokenizer is available, if not download it
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("OPEN_AI_SECRET_KEY")
if not api_key:
    raise ValueError("OPEN_AI_SECRET_KEY environment variable is not set")

# Loop over all files in the data folder
data_folder = "data"
if not os.path.exists(data_folder):
    logger.error(f"Data folder not found: {data_folder}")
    raise FileNotFoundError(f"Data folder not found: {data_folder}")

def load_documents(data_folder):
    all_docs = []
    for filename in os.listdir(data_folder):
        file_path = os.path.join(data_folder, filename)
        try:
            if filename.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                all_docs.extend(docs)
                logger.info(f"Loaded PDF file: {filename}")
            elif filename.lower().endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
                all_docs.extend(docs)
                logger.info(f"Loaded DOCX file: {filename}")
            else:
                logger.warning(f"Skipping unsupported file: {filename}")
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
    return all_docs

# Chunking strategy: fixed size with overlap
def split_by_fixed_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    logger.info(f"Split documents into {len(chunks)} fixed-size chunks")
    return chunks

# Chunking strategy: sentence-based
def split_by_sentences(documents):
    all_chunks = []
    for doc in documents:
        sentences = sent_tokenize(doc.page_content)
        for sentence in sentences:
            all_chunks.append(doc.__class__(page_content=sentence))
    logger.info(f"Split documents into {len(all_chunks)} sentence chunks")
    return all_chunks

# Chunking strategy: paragraph-based
def split_by_paragraphs(documents):
    all_chunks = []
    splitter = CharacterTextSplitter(separator="\n\n", chunk_size=500, chunk_overlap=100)
    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            all_chunks.append(Document(page_content=chunk))
    logger.info(f"Split documents into {len(all_chunks)} paragraph chunks")
    return all_chunks

def create_embeddings_and_store(docs, api_key, save_path="vector_store"):
    embedding = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.from_documents(docs, embedding)
    db.save_local(save_path)
    logger.info(f"Saved vector store to {save_path}")
    return db

#Loads documents, splits text, creates embeddings, and saves vector store.
# Supports different chunking strategies, Default chunking is paragraphs
def process_documents(data_folder, chunking_strategy="paragraphs"):
    logger.info(f"Starting document processing using '{chunking_strategy}' chunking strategy...")
    all_docs = load_documents(data_folder)
    if not all_docs:
        logger.error("No documents found in the specified folder.")
        raise ValueError("No documents found in the specified folder.")
    
    logger.info(f"Loaded {len(all_docs)} documents. Splitting into chunks...")

    if chunking_strategy == "fixed":
        docs = split_by_fixed_chunks(all_docs)
    elif chunking_strategy == "sentences":
        docs = split_by_sentences(all_docs)
    elif chunking_strategy == "paragraphs":
        docs = split_by_paragraphs(all_docs)
    else:
        logger.error(f"Invalid chunking strategy selected: {chunking_strategy}")
        raise ValueError("Invalid chunking strategy selected.")

    if not docs:
        logger.error("No documents created after splitting.")
        raise ValueError("No documents created after splitting.")

    logger.info(f"Total chunks created: {len(docs)}")
    logger.info("Creating embeddings and storing them in the vector database...")
    
    # Create embeddings and store them in a vector store
    db = create_embeddings_and_store(docs, api_key)
    logger.info("Document processing complete.")
    return db

#Just for testing purposes
if __name__ == "__main__":
    data_folder = "./data"  # כאן שמים את הקבצים שברצונך לעבד
    db = process_documents(data_folder, chunking_strategy="paragraphs")

    query = "Explain to me what a linear equation is?"
    results = db.similarity_search(query, k=3)

    for i, doc in enumerate(results):
        logger.info(f"--- Result {i+1} ---")
        logger.info(doc.page_content)
        logger.info("")