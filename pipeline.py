import os
from ingestion.document_loader import DocumentLoader
from ingestion.text_processor import TextProcessor
from ingestion.chunker import Chunker
from ingestion.indexer import Indexer

# Définition des chemins
RAW_DOCS_DIR = "data/documents"
PROCESSED_DIR = "data/processed/chunks"
VECTOR_DB_PATH = "data/vector_store/chroma_db"

def run_ingestion():
    # 1️⃣ Charger les documents PDF
    loader = DocumentLoader(RAW_DOCS_DIR)
    documents = loader.load_pdfs()

    # 2️⃣ Nettoyer et prétraiter le texte
    processor = TextProcessor()
    cleaned_documents = {name: processor.clean_text(text) for name, text in documents.items()}

    # 3️⃣ Segmenter les documents
    chunker = Chunker()
    chunked_documents = {name: chunker.chunk_text(text) for name, text in cleaned_documents.items()}

    # 4️⃣ Sauvegarder les chunks sous forme de fichiers texte
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    for name, chunks in chunked_documents.items():
        with open(os.path.join(PROCESSED_DIR, f"{name}.txt"), "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(chunk + "\n\n")

    # 5️⃣ Indexer les chunks dans ChromaDB
    indexer = Indexer(VECTOR_DB_PATH)
    for name, chunks in chunked_documents.items():
        indexer.index_chunks(chunks, name)

    print("✅ Ingestion et indexation terminées avec succès !")

if __name__ == "__main__":
    run_ingestion()
