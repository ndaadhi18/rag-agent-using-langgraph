import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_FOLDER = "data/"
VECTOR_STORE_PATH = "local_vector_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


def load_documents(data_folder: str) -> list:

    documents = []
    data_path = Path(data_folder)
    
    print(f"\n{'='*60}")
    print("Loading documents...")
    print(f"{'='*60}\n")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data folder not found: {data_folder}")
    
    files = list(data_path.glob("*"))
    
    for file_path in files:
        if file_path.is_file():
            try:
                # Load .txt files
                if file_path.suffix == ".txt":
                    print(f"Loading TXT: {file_path.name}")
                    loader = TextLoader(str(file_path), encoding="utf-8")
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"Loaded {len(docs)} document(s) from {file_path.name}")
                
                # Load .pdf files
                elif file_path.suffix == ".pdf":
                    print(f"Loading PDF: {file_path.name}")
                    loader = PyPDFLoader(str(file_path))
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"Loaded {len(docs)} page(s) from {file_path.name}")
                    
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
    
    print(f"\nTotal documents loaded: {len(documents)}\n")
    return documents


def chunk_documents(documents, chunk_size: int, chunk_overlap: int) -> list:

    print(f"{'='*60}")
    print("Chunking documents...)")
    print(f"{'='*60}\n")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    print(f"Created {len(chunks)} chunks from {len(documents)} documents\n")
    
    return chunks


def create_vector_store(chunks, embedding_model: str, persist_directory: str) -> Chroma:

    print(f"{'='*60}")
    print(f"Creating embeddings using: {embedding_model}")
    print(f"{'='*60}\n")
    
    # Initialize embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print(f"Storing vectors in: {persist_directory + '/'}\n")
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"Vector store created with {len(chunks)} embeddings\n")
    
    return vectorstore


def main():
    """Main ingestion pipeline"""
    print("\n" + "="*60)
    print("STARTING DOCUMENT INGESTION PIPELINE")
    print("="*60 + "\n")
    
    try:
        # Step 1: Load documents
        documents = load_documents(DATA_FOLDER)
        
        if not documents:
            print("No documents found. Please add .txt or .pdf files to the Data folder.")
            return
        
        # Step 2: Chunk documents
        chunks = chunk_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
        
        # Step 3: Create vector store
        vectorstore= create_vector_store(chunks, EMBEDDING_MODEL, VECTOR_STORE_PATH)
        
        # Final summary
        print("="*60)
        print("INGESTION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nSummary:")
        print(f"Documents processed: {len(documents)}")
        print(f"Chunks created: {len(chunks)}")
        print(f"Vector store location: {Path(VECTOR_STORE_PATH).absolute()}")
        
    except Exception as e:
        print(f"\nError during ingestion: {e}")
        raise


if __name__ == "__main__":
    main()
