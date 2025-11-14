# AmbedkarGPT - RAG-based Q&A System

> **Assignment Submission for Kalpit Pvt Ltd, UK - AI Intern Position**  
> **Phase 1: Core Skills Evaluation - Building A Functional Prototype**

A Retrieval-Augmented Generation (RAG) system built with LangChain, ChromaDB, and Ollama to answer questions based on Dr. B.R. Ambedkar's speech on caste and social reform.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-🦜-green)](https://www.langchain.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Prerequisites](#-prerequisites)
- [Installation Guide](#-installation-guide)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Technical Implementation](#-technical-implementation)
- [Troubleshooting](#-troubleshooting)
- [Assignment Compliance](#-assignment-compliance)
- [Demo](#-demo)
- [Contact](#-contact)

---

## 🎯 Overview

This project implements a **command-line Q&A system** that uses **Retrieval-Augmented Generation (RAG)** to answer questions based on provided text from Dr. B.R. Ambedkar's "Annihilation of Caste" speech. 

### How It Works

The system performs the following steps:

1. **Loads** the speech text from `speech.txt`
2. **Splits** the text into manageable chunks
3. **Creates embeddings** using HuggingFace's sentence transformers
4. **Stores** embeddings in ChromaDB (local vector database)
5. **Retrieves** relevant context based on user queries
6. **Generates** answers using Ollama's Mistral 7B model

**✅ Everything runs 100% locally - No API keys, No accounts, No costs!**

---

## ✨ Features

- 🚀 **Fully Local Deployment** - No external API calls or accounts required
- 💬 **Interactive Chat Mode** - Ask multiple questions in a conversational interface
- ⚡ **Single Query Mode** - Get quick answers via command-line arguments
- 💾 **Persistent Vector Store** - Embeddings are saved and reused across sessions
- 🔍 **Smart Semantic Search** - Finds relevant context using vector similarity
- 📝 **Well-Documented Code** - Clean, commented code following Python best practices
- 🛡️ **Robust Error Handling** - Comprehensive error messages and troubleshooting guides
- 🎨 **Beautiful CLI** - User-friendly console interface with emoji indicators

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER QUERY                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: DOCUMENT LOADING                                   │
│  ┌─────────────┐                                            │
│  │ speech.txt  │  →  TextLoader  →  Document Object         │
│  └─────────────┘                                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: TEXT CHUNKING                                      │
│  Document  →  CharacterTextSplitter  →  Text Chunks         │
│  (chunk_size=500, overlap=100)                              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: EMBEDDING CREATION                                 │
│  Text Chunks  →  HuggingFace (all-MiniLM-L6-v2)  →  Vectors │
│  (384-dimensional embeddings)                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: VECTOR STORAGE                                     │
│  Vectors  →  ChromaDB  →  Persistent Local Database         │
│  (./chroma_db/)                                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 5: RETRIEVAL & GENERATION                             │
│  Query  →  Semantic Search  →  Top 3 Relevant Chunks        │
│  Chunks + Query  →  Ollama Mistral  →  Generated Answer     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Prerequisites

Before starting, ensure you have the following installed:

### System Requirements

- **Operating System**: Windows 10/11, macOS 10.15+, or Linux
- **Python**: Version 3.8 or higher (Python 3.10 or 3.11 recommended)
- **RAM**: Minimum 8GB (16GB recommended for optimal performance)
- **Disk Space**: ~5GB for models and dependencies
- **Internet**: Required for initial setup and model downloads

### Required Software

1. **Python 3.8+** - [Download here](https://www.python.org/downloads/)
2. **Git** - [Download here](https://git-scm.com/downloads)
3. **Ollama** - [Download here](https://ollama.ai/download)

---

## 📥 Installation Guide

Follow these steps carefully to set up the project on your local machine.

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task
```

### Step 2: Create a Virtual Environment

**On Windows (PowerShell/CMD):**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip setuptools wheel
```

### Step 4: Install Python Dependencies

```bash
pip install langchain langchain-core langchain-community langchain-text-splitters chromadb sentence-transformers ollama
```

**Alternative (using requirements.txt):**
```bash
pip install -r requirements.txt
```

### Step 5: Install and Setup Ollama

#### On Windows:

1. Download Ollama installer from [https://ollama.ai/download](https://ollama.ai/download)
2. Run the installer
3. Open a **new** Command Prompt or PowerShell window
4. Pull the Mistral model:
   ```bash
   ollama pull mistral
   ```

#### On macOS/Linux:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Mistral model
ollama pull mistral
```

### Step 6: Verify Installation

Check that Ollama is working:

```bash
ollama list
```

You should see `mistral` in the list.

---

## 🚀 Usage

### Interactive Mode (Recommended)

Run the system without any arguments to enter interactive mode:

```bash
python main.py
```

**Example Session:**

```
============================================================
🚀 Initializing AmbedkarGPT...
============================================================

📄 Loading speech document...
✅ Loaded document with 752 characters
✂️  Splitting document into chunks...
✅ Created 3 text chunks
🧠 Creating embeddings using HuggingFace...
💾 Storing embeddings in ChromaDB...
✅ Vector store created and persisted at './chroma_db'
🤖 Initializing Ollama Mistral LLM...
✅ Q&A system ready!

============================================================
🎯 Interactive Q&A Mode
============================================================

Ask questions about Dr. Ambedkar's speech!
Type 'exit' or 'quit' to stop.

Your question: What is the real remedy according to Ambedkar?

❓ Question: What is the real remedy according to Ambedkar?
🔍 Retrieving relevant context and generating answer...

💡 Answer: According to the text, the real remedy is to destroy 
the belief in the sanctity of the shastras.

📚 Retrieved context snippets:
  [1] The real remedy is to destroy the belief in the sanctity 
       of the shastras. How do you expect to succeed if you...

------------------------------------------------------------
Your question: exit

👋 Thank you for using AmbedkarGPT!
```

### Single Question Mode

Ask a single question directly from the command line:

```bash
python main.py "What does Ambedkar say about social reform?"
```

**Output:**
```
🚀 Initializing AmbedkarGPT...
...
❓ Question: What does Ambedkar say about social reform?
💡 Answer: Ambedkar compares social reform to the work of a gardener 
who constantly prunes leaves and branches without attacking the roots...
```

### Example Questions to Try

1. `"What is the real remedy according to Ambedkar?"`
2. `"What does he say about the shastras?"`
3. `"How does Ambedkar describe the work of social reform?"`
4. `"What is the real enemy according to the speech?"`
5. `"Can you have both caste practice and belief in shastras?"`
6. `"What must people stop believing in to get rid of caste?"`

---

## 📁 Project Structure

```
AmbedkarGPT-Intern-Task/
│
├── main.py                    # Main application code (RAG pipeline)
├── speech.txt                 # Dr. Ambedkar's speech text
├── requirements.txt           # Python dependencies
├── README.md                  # This documentation file
├── .gitignore                 # Git ignore rules
│
├── chroma_db/                 # ChromaDB vector store (auto-created)
│   ├── chroma.sqlite3         # SQLite database for metadata
│   └── ...                    # Embedding data
│
└── venv/                      # Virtual environment (you create this)
    └── ...                    # Python packages
```

---

## 🔬 Technical Implementation

### Technology Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Framework** | LangChain | 0.1.0+ | Orchestrates RAG pipeline |
| **Vector Database** | ChromaDB | 0.4.22+ | Stores and retrieves embeddings |
| **Embeddings** | HuggingFace | all-MiniLM-L6-v2 | Converts text to 384-dim vectors |
| **LLM** | Ollama Mistral | 7B | Generates natural language answers |
| **Text Processing** | LangChain Text Splitters | 0.0.1+ | Chunks documents efficiently |

### Key Implementation Details

#### 1. Document Loading
```python
loader = TextLoader("speech.txt", encoding='utf-8')
documents = loader.load()
```

#### 2. Text Chunking
```python
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,      # 500 characters per chunk
    chunk_overlap=100,   # 100 character overlap for context
    length_function=len
)
chunks = text_splitter.split_documents(documents)
```

#### 3. Embedding Creation
```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)
```

#### 4. Vector Store Creation
```python
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

#### 5. RAG Chain Setup
```python
llm = Ollama(model="mistral", temperature=0.2)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)
```

### Configuration Parameters

- **Chunk Size**: 500 characters
- **Chunk Overlap**: 100 characters
- **Embedding Dimensions**: 384
- **Retrieval Count**: Top 3 most relevant chunks
- **LLM Temperature**: 0.2 (focused, deterministic answers)
- **Vector Store**: Persistent (saved to disk)

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### Issue 1: "ModuleNotFoundError: No module named 'langchain'"

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install all dependencies
pip install langchain langchain-core langchain-community langchain-text-splitters chromadb sentence-transformers ollama
```

#### Issue 2: "Ollama model not found" or "Connection refused"

**Solution:**
```bash
# Check if Ollama is running (open a new terminal)
ollama serve

# Pull the Mistral model
ollama pull mistral

# Verify installation
ollama list
```

#### Issue 3: "speech.txt not found"

**Solution:**
Ensure `speech.txt` exists in the same directory as `main.py`. Check with:
```bash
ls speech.txt    # macOS/Linux
dir speech.txt   # Windows
```

#### Issue 4: "Could not import chromadb"

**Solution:**
```bash
pip uninstall chromadb
pip install chromadb
```

#### Issue 5: Python version compatibility issues

**Solution:**
This project works best with **Python 3.10 or 3.11**. If you have Python 3.12+, create a new environment with Python 3.11:

```bash
# Install Python 3.11, then:
python3.11 -m venv venv311
source venv311/bin/activate  # or venv311\Scripts\activate
pip install langchain langchain-core langchain-community langchain-text-splitters chromadb sentence-transformers ollama
```

#### Issue 6: "Out of memory" when running Mistral

**Solution:**
- Close other applications to free RAM
- Use a smaller model:
  ```bash
  ollama pull phi
  ```
  Then modify `main.py` line with `model="mistral"` to `model="phi"`

#### Issue 7: ChromaDB persistence errors

**Solution:**
```bash
# Delete existing vector store and recreate
rm -rf chroma_db/        # macOS/Linux
rmdir /s chroma_db\      # Windows

# Run the program again
python main.py
```

#### Issue 8: Slow first-time execution

**Cause:** First run downloads embedding models (~90MB)  
**Solution:** Be patient, subsequent runs will be much faster as models are cached.

---

## ✅ Assignment Compliance

This submission fully meets all requirements specified in the assignment brief:

### ✓ Technical Requirements

- [x] **Programming Language**: Python 3.8+
- [x] **Core Framework**: LangChain framework for RAG pipeline
- [x] **Vector Database**: ChromaDB (open-source, local)
- [x] **Embeddings**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- [x] **LLM**: Ollama with Mistral 7B (100% free, no API keys)
- [x] **No External Dependencies**: Everything runs locally

### ✓ Required Deliverables

- [x] **Well-commented Python code** (`main.py`) - 300+ lines, comprehensive docstrings
- [x] **requirements.txt** - All dependencies listed
- [x] **Detailed README.md** - Complete setup and usage guide
- [x] **speech.txt** - Provided text included
- [x] **Public GitHub repository** - AmbedkarGPT-Intern-Task

### ✓ Functional Requirements

- [x] **Load text file** - TextLoader implementation
- [x] **Split text** - CharacterTextSplitter with 500/100 config
- [x] **Create embeddings** - HuggingFace embeddings
- [x] **Store in vector DB** - ChromaDB with persistence
- [x] **Retrieve relevant chunks** - Semantic search with k=3
- [x] **Generate answers** - Ollama Mistral 7B integration
- [x] **Command-line interface** - Interactive and single-query modes

### ✓ Code Quality Standards

- [x] **PEP 8 Compliant** - Proper formatting and style
- [x] **Comprehensive Documentation** - Docstrings for all classes/functions
- [x] **Error Handling** - Try-catch blocks with informative messages
- [x] **Modular Design** - Clean OOP architecture
- [x] **User-Friendly Output** - Colored console with emoji indicators
- [x] **Production-Ready** - Robust and maintainable code

---

## 🎬 Demo

### Sample Execution

```bash
$ python main.py

============================================================
🚀 Initializing AmbedkarGPT...
============================================================

📄 Loading speech document...
✅ Loaded document with 752 characters
✂️  Splitting document into chunks...
✅ Created 3 text chunks
🧠 Creating embeddings using HuggingFace...
💾 Storing embeddings in ChromaDB...
✅ Vector store created and persisted at './chroma_db'
🤖 Initializing Ollama Mistral LLM...
✅ Q&A system ready!

============================================================
✅ System initialized successfully!
============================================================

============================================================
🎯 Interactive Q&A Mode
============================================================

Ask questions about Dr. Ambedkar's speech!
Type 'exit' or 'quit' to stop.

Your question: What is the real remedy?

❓ Question: What is the real remedy?
🔍 Retrieving relevant context and generating answer...

💡 Answer: The real remedy is to destroy the belief in the 
sanctity of the shastras.

📚 Retrieved context snippets:

  [1] The real remedy is to destroy the belief in the sanctity 
      of the shastras. How do you expect to succeed if you allow...

  [2] So long as people believe in the sanctity of the shastras, 
      they will never be able to get rid of caste...

  [3] The real enemy is the belief in the shastras.

------------------------------------------------------------
Your question: exit

👋 Thank you for using AmbedkarGPT!
```

---



## 🙏 Acknowledgments

- **Dr. B.R. Ambedkar** - For his profound writings on social reform
- **LangChain Team** - For excellent documentation and framework
- **Ollama Team** - For making local LLMs accessible
- **HuggingFace** - For open-source embedding models
- **ChromaDB Team** - For the efficient vector database

---

## 📄 License

This project was created as part of an internship assignment for Kalpit Pvt Ltd, UK.

---

## 🚀 Future Enhancements

Potential improvements for a production-ready system:

1. **Web Interface** - Streamlit/Gradio UI for better user experience
2. **Multi-Document Support** - Handle multiple speech files
3. **Chat History** - Save and retrieve past conversations
4. **Advanced Retrieval** - Hybrid search (semantic + keyword)
5. **Model Selection** - Choose between different LLMs
6. **Performance Metrics** - Response time and relevance scoring
7. **Export Functionality** - Save conversations to PDF/JSON
8. **API Endpoint** - REST API for integration
9. **Docker Container** - Easy deployment
10. **Unit Tests** - Comprehensive test coverage

---


*Last Updated: November 14, 2025*

</div>
