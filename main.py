from langchain.document_loaders import PyPDFLoader,Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,SentenceSplitter, CharacterTextSplitter 
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
data_folder = "data"

api_key = os.getenv("OPEN_AI_SECRET_KEY")
if not api_key:
    raise ValueError("OPEN_AI_SECRET_KEY environment variable is not set")

# Loop over all files in the data folder, load documents from folder
def load_documents(data_folder):
    all_docs = []
    try:
        for filename in os.listdir(data_folder):
            file_path = os.path.join(data_folder, filename)
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                all_docs.extend(documents)
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                documents = loader.load()
                all_docs.extend(documents)
    except Exception as e:
        print(f"Error loading files: {e}")
    return all_docs

# Chunking strategy: fixed size with overlap
def split_by_fixed_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(documents)

# Chunking strategy: sentence-based
def split_by_sentences(documents):
    splitter = SentenceSplitter()
    return splitter.split_documents(documents)

# Chunking strategy: paragraph-based
def split_by_paragraphs(documents):
    text = "\n".join([doc.page_content for doc in documents])
    splitter = CharacterTextSplitter(separator="\n\n", chunk_size=500, chunk_overlap=100)
    return splitter.create_documents([text])

# Choose which strategy to apply
chunking_strategy = "paragraphs" #options: fixed, sentences, paragraphs
if chunking_strategy not in ["fixed", "sentences", "paragraphs"]:
    raise ValueError("Invalid chunking strategy selected.")

print(f"Using chunking strategy: {chunking_strategy}")
all_docs = load_documents(data_folder)

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

#create embadding and vector data base
embedding = OpenAIEmbeddings(openai_api_key=api_key)
db = FAISS.from_documents(docs, embedding)
db.save_local("vector_store")

print("Files have been processed and saved.")

# Perform a query on the vector database
query = "Explain to me what a linear equation is?"

# Ranks the results by semantic similarity and returns the most similar segments
result = db.similarity_search(query, k=3)  # Search for the 3 closest results

# Loop to print the 3 most relevant answers
for i, doc in enumerate(result):
    print(f"--- Result {i+1} ---")
    print(doc.page_content)
    print()
