# 🤖 AIbot

Welcome to *AIbot* — an intelligent document-processing tool that uses OpenAI embeddings combined with FAISS to build a fast, reliable, and scalable semantic search engine for your PDF and DOCX files.

---

## 📚 What It Does

- 🧠 Generates high-quality embeddings from documents via OpenAI’s API
- 📄 Supports PDF input files
- ✂️ Implements flexible text chunking strategies (fixed size, sentence, paragraph)
- ⚡ Efficiently stores and retrieves vectors using FAISS for fast semantic search
---

## 🛠️ Requirements

- Python 3.7+
- An OpenAI API Key (with available quota)

---

## 🚀 Setup

### 1. Clone the repository and move into the project directory:

```bash
git clone https://github.com/tomerdolev/AIbot.git
cd AIbot
```
### 2. Set up your API key:
Create a file called .env in the root directory and add your OpenAI API key:

```bash
OPEN_AI_SECRET_KEY=your_openai_key_here
```

---

## 📦 Usage

To process documents and build a vector store:

Place your .pdf or .docx files in the ./data folder.

Run the main script:
```bash
python main.py
```

This will:

- Load your documents
- Split them based on the selected strategy (default: paragraph)
- Generate embeddings
- Save them in a FAISS vector store

  ---
  
## 📂 Project Structure

```bash
AIbot/
├── data/                 # Folder with your input PDF/DOCX files
├── vector_store/         # FAISS index will be saved here
├── main.py               # Main script to process documents and query
├── .env                  # File with your OpenAI API key
├── requirements.txt      # Python dependencies
└── README.md             # Project overview
```

## 🔮 Future Improvements

- Add support for additional file types (e.g., TXT, HTML)
- Add a web interface (Streamlit or Gradio)
- Enable persistent querying across sessions
- Integrate metadata for document filtering (e.g., titles, dates)


