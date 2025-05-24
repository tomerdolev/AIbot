import os
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize
from langchain.document_loaders import PyPDFLoader,Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter 
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# Ensure necessary NLTK resources are downloaded ( for sentence splitting)
nltk.download('punkt')

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("OPEN_AI_SECRET_KEY")
if not api_key:
    raise ValueError("OPEN_AI_SECRET_KEY environment variable is not set")

# Loop over all files in the data folder
data_folder = "data"
def load_documents(data_folder):
    all_docs = []
    for filename in os.listdir(data_folder):
        file_path = os.path.join(data_folder, filename)
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                all_docs.extend(docs)
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
                all_docs.extend(docs)
            else:
                print(f"Skipping unsupported file: {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    return all_docs

# Chunking strategy: fixed size with overlap
def split_by_fixed_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(documents)

# Chunking strategy: sentence-based
def split_by_sentences(documents):
    all_chunks = []
    for doc in documents:
        sentences = sent_tokenize(doc.page_content)
        for sentence in sentences:
            all_chunks.append(doc.__class__(page_content=sentence))
    return all_chunks

# Chunking strategy: paragraph-based
def split_by_paragraphs(documents):
    all_chunks = []
    splitter = CharacterTextSplitter(separator="\n\n", chunk_size=500, chunk_overlap=100)
    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        for chunk in chunks:
            all_chunks.append(Document(page_content=chunk))
    return all_chunks

def create_embeddings_and_store(docs, api_key, save_path="vector_store"):
    embedding = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.from_documents(docs, embedding)
    db.save_local(save_path)
    return db

#Loads documents, splits text, creates embeddings, and saves vector store.
# Supports different chunking strategies, Default chunking is paragraphs
def process_documents(data_folder, chunking_strategy="paragraphs"):
    all_docs = load_documents(data_folder)
    if not all_docs:
        raise ValueError("No documents found in the specified folder.")
    
    if chunking_strategy == "fixed":
        docs = split_by_fixed_chunks(all_docs)
    elif chunking_strategy == "sentences":
        docs = split_by_sentences(all_docs)
    elif chunking_strategy == "paragraphs":
        docs = split_by_paragraphs(all_docs)
    else:
        raise ValueError("Invalid chunking strategy selected.")

    if not docs:
        raise ValueError("No documents created after splitting.")

    # Create embeddings and store them in a vector store
    db = create_embeddings_and_store(docs, api_key)
    return db

#Just for testing purposes
if __name__ == "__main__":
    data_folder = "./data"  # כאן שמים את הקבצים שברצונך לעבד
    db = process_documents(data_folder, chunking_strategy="paragraphs")

    query = "Explain to me what a linear equation is?"
    results = db.similarity_search(query, k=3)

    for i, doc in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(doc.page_content)
        print()