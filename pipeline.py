import os
import json
from ingestion.document_loader import DocumentLoader
from ingestion.text_processor import TextProcessor
from ingestion.chunker import Chunker
from ingestion.indexer import Indexer

RAW_DOCS_DIR = "data/documents"
PROCESSED_DIR = "data/processed/chunks"
VECTOR_DB_PATH = "data/vector_store/chroma_db"
TRACK_FILE = "data/processed/ingested_files.json"

def load_previous_ingestions():
    if os.path.exists(TRACK_FILE):
        with open(TRACK_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_ingested_files(filenames):
    with open(TRACK_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(filenames), f, indent=2)

def run_ingestion():
    print("\n🚀 [0] DÉMARRAGE DU PIPELINE D'INGESTION...\n")

    # 🔍 Étape 1 : Vérifier les nouveaux fichiers PDF
    all_files = set(f for f in os.listdir(RAW_DOCS_DIR) if f.endswith(".pdf"))
    already_ingested = load_previous_ingestions()
    new_files = all_files - already_ingested

    if not new_files:
        print("✅ Aucun nouveau document à ingérer. Tous les fichiers ont déjà été traités.\n")
        return

    print(f"📂 [1] {len(new_files)} nouveau(x) fichier(s) détecté(s) :")
    for f in new_files:
        print(f"   └─ {f}")

    # 📥 Étape 2 : Charger les nouveaux fichiers PDF
    loader = DocumentLoader(RAW_DOCS_DIR)
    all_loaded_docs = loader.load_pdfs()
    documents = {name: text for name, text in all_loaded_docs.items() if name in new_files}
    print(f"\n📥 [2] {len(documents)} document(s) PDF chargé(s) avec succès.")

    # 🧹 Étape 3 : Nettoyage du texte
    processor = TextProcessor()
    cleaned_documents = {}
    for name, text in documents.items():
        cleaned_documents[name] = processor.clean_text(text)
        print(f"   ✔️ Texte nettoyé pour : {name}")

    # ✂️ Étape 4 : Découpage en chunks
    chunker = Chunker()
    chunked_documents = {}
    for name, text in cleaned_documents.items():
        chunks = chunker.chunk_text(text)
        chunked_documents[name] = chunks
        print(f"   📎 {len(chunks)} chunk(s) généré(s) pour : {name}")

    # 💾 Étape 5 : Sauvegarde locale des chunks
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    for name, chunks in chunked_documents.items():
        path = os.path.join(PROCESSED_DIR, f"{name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(chunk + "\n\n")
        print(f"   💾 Chunks sauvegardés dans : {path}")

    # 🧠 Étape 6 : Indexation dans ChromaDB
    indexer = Indexer(VECTOR_DB_PATH)
    for name, chunks in chunked_documents.items():
        indexer.index_chunks(chunks, name)
        print(f"   🧠 {len(chunks)} chunk(s) indexé(s) pour : {name}")

    # 📝 Étape 7 : Mise à jour de l’historique des fichiers traités
    save_ingested_files(all_files)
    print("\n📜 Historique des fichiers mis à jour.")

    # ✅ Fin du pipeline
    print("\n🎉 INGESTION TERMINÉE AVEC SUCCÈS ! Tous les nouveaux fichiers ont été traités et indexés.\n")

if __name__ == "__main__":
    run_ingestion()
