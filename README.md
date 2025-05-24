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

## 📦 Usage

To process documents and build a vector store:

Place your PDF and DOCX files inside the ./data folder.

Run the main script:
```bash
python main.py

```

## 🚀 Setup

### 1. Clone the repository and move into the project directory:

```bash
git clone https://github.com/tomerdolev/AIbot.git
cd AIbot
```

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

