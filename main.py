"""
AmbedkarGPT - A RAG-based Q&A System
Built for Kalpit Pvt Ltd, UK - AI Intern Assignment

This system implements a Retrieval-Augmented Generation (RAG) pipeline
to answer questions based on Dr. B.R. Ambedkar's speech text.
"""

import os
import sys
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate


class AmbedkarGPT:
    """
    Main class for the RAG-based Q&A system.
    Handles document loading, vector store creation, and query processing.
    """
    
    def __init__(self, speech_file="speech.txt", persist_directory="./chroma_db"):
        """
        Initialize the AmbedkarGPT system.
        
        Args:
            speech_file (str): Path to the speech text file
            persist_directory (str): Directory to persist the Chroma vector store
        """
        self.speech_file = speech_file
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.qa_chain = None
        
    def load_and_split_documents(self):
        """
        Step 1 & 2: Load the speech text and split it into chunks.
        
        Returns:
            list: List of document chunks
        """
        print("📄 Loading speech document...")
        
        # Check if file exists
        if not os.path.exists(self.speech_file):
            raise FileNotFoundError(f"Speech file '{self.speech_file}' not found!")
        
        # Load the document
        loader = TextLoader(self.speech_file, encoding='utf-8')
        documents = loader.load()
        
        print(f"✅ Loaded document with {len(documents[0].page_content)} characters")
        
        # Split into chunks
        print("✂️  Splitting document into chunks...")
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        print(f"✅ Created {len(chunks)} text chunks")
        return chunks
    
    def create_vector_store(self, chunks):
        """
        Step 3: Create embeddings and store them in ChromaDB.
        
        Args:
            chunks (list): List of document chunks
        """
        print("🧠 Creating embeddings using HuggingFace...")
        
        # Initialize HuggingFace embeddings (runs locally, no API key needed)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        print("💾 Storing embeddings in ChromaDB...")
        
        # Create and persist the vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.persist_directory
        )
        
        print(f"✅ Vector store created and persisted at '{self.persist_directory}'")
    
    def setup_qa_chain(self):
        """
        Step 4 & 5: Set up the retrieval and LLM chain.
        Creates a RetrievalQA chain that retrieves relevant chunks
        and generates answers using Ollama Mistral.
        """
        print("🤖 Initializing Ollama Mistral LLM...")
        
        # Initialize Ollama with Mistral model (runs locally)
        llm = Ollama(
            model="mistral",
            temperature=0.2,  # Lower temperature for more focused answers
        )
        
        # Create a custom prompt template for better answers
        prompt_template = """You are an AI assistant specialized in answering questions about Dr. B.R. Ambedkar's speech on caste and social reform.

Use the following context to answer the question. If you cannot find the answer in the context, say "I cannot find this information in the provided speech."

Context: {context}

Question: {question}

Answer (be concise and based only on the context):"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create the RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}  # Retrieve top 3 relevant chunks
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        print("✅ Q&A system ready!")
    
    def initialize(self):
        """
        Complete initialization of the system.
        Loads documents, creates vector store, and sets up QA chain.
        """
        print("\n" + "="*60)
        print("🚀 Initializing AmbedkarGPT...")
        print("="*60 + "\n")
        
        # Check if vector store already exists
        if os.path.exists(self.persist_directory):
            print("📦 Found existing vector store, loading...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embeddings
            )
            print("✅ Vector store loaded successfully")
        else:
            # Load and process documents
            chunks = self.load_and_split_documents()
            
            # Create vector store
            self.create_vector_store(chunks)
        
        # Setup QA chain
        self.setup_qa_chain()
        
        print("\n" + "="*60)
        print("✅ System initialized successfully!")
        print("="*60 + "\n")
    
    def ask(self, question):
        """
        Ask a question and get an answer.
        
        Args:
            question (str): The question to ask
            
        Returns:
            str: The generated answer
        """
        if self.qa_chain is None:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        print(f"\n❓ Question: {question}")
        print("🔍 Retrieving relevant context and generating answer...\n")
        
        # Get answer from the QA chain
        result = self.qa_chain.invoke({"query": question})
        answer = result['result']
        
        print(f"💡 Answer: {answer}\n")
        
        # Optionally show source documents
        if result.get('source_documents'):
            print("📚 Retrieved context snippets:")
            for i, doc in enumerate(result['source_documents'], 1):
                print(f"\n  [{i}] {doc.page_content[:200]}...")
        
        return answer


def interactive_mode(gpt_system):
    """
    Run the system in interactive mode for continuous Q&A.
    
    Args:
        gpt_system (AmbedkarGPT): Initialized AmbedkarGPT instance
    """
    print("\n" + "="*60)
    print("🎯 Interactive Q&A Mode")
    print("="*60)
    print("\nAsk questions about Dr. Ambedkar's speech!")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            question = input("Your question: ").strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Thank you for using AmbedkarGPT!")
                break
            
            if not question:
                print("⚠️  Please enter a question.")
                continue
            
            gpt_system.ask(question)
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")


def main():
    """
    Main entry point of the application.
    """
    try:
        # Create and initialize the system
        gpt_system = AmbedkarGPT()
        gpt_system.initialize()
        
        # Check if a question was provided as command-line argument
        if len(sys.argv) > 1:
            # Single question mode
            question = " ".join(sys.argv[1:])
            gpt_system.ask(question)
        else:
            # Interactive mode
            interactive_mode(gpt_system)
            
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Make sure 'speech.txt' exists in the current directory.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Ensure Ollama is installed and running: ollama serve")
        print("2. Ensure Mistral model is pulled: ollama pull mistral")
        print("3. Check that all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()