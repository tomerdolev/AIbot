# Import tools from langchain library
# Here I defined all the tools we need: loading files, splitting texts, storing and creating embeddings
from langchain.document_loaders import PyPDFLoader,Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add all loaded documents to a list
# Scan the data folder, all found documents are saved in a large list of texts
# I separated between docs and PDF files
all_docs = []
data_folder = "data"

# Loop over all files in the data folder
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

# Splits the segments into uniform size and defined overlap
# Allows us to create embeddings optimally
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(all_docs)

# Create embeddings
# Each segment is converted to a numeric vector that describes its meaning in natural language
# Here I used my OPENAI API key
api_key = os.getenv("OPEN_AI_SECRET_KEY")
if not api_key:
    raise ValueError("OPEN_AI_SECRET_KEY environment variable is not set")
    
embedding = OpenAIEmbeddings(openai_api_key=api_key)
# Save the vectors in a FAISS database
# This will serve as our vector search engine for fast retrieval of similar texts
db = FAISS.from_documents(docs, embedding)
db.save_local("vector_store")

# This message is for me, just to verify that everything is working up to this point
print("Files have been processed and saved.")

# Perform a query on the vector database
query = "Explain to me what a linear equation is?"

# similarity_search converts the query to embeddings behind the scenes
# It compares the similarity between the vectors in the database and the query after converting it to embeddings
# Ranks the results by semantic similarity and returns the most similar segments

docs = db.similarity_search(query, k=3)  # Search for the 3 closest results

# Loop to print the 3 most relevant answers
for i, doc in enumerate(docs):
    print(f"--- Result {i+1} ---")
    print(doc.page_content)
    print()
