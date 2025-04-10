import os
import shutil
import json
from typing import List, Any, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import DOCUMENTS_DIR, VECTOR_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
from utils import logger, get_document_files

INGESTED_FILES_RECORD = os.path.join(VECTOR_DB_DIR, ".ingested_files.json")

class DocumentIngestion:
    def __init__(self,
                 documents_dir: str = DOCUMENTS_DIR,
                 vector_db_dir: str = VECTOR_DB_DIR,
                 chunk_size: int = 2000,  # élargir les chunks
                 chunk_overlap: int = CHUNK_OVERLAP,
                 embedding_model: str = EMBEDDING_MODEL):
        
        self.documents_dir = documents_dir
        self.vector_db_dir = vector_db_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        os.makedirs(self.documents_dir, exist_ok=True)
        os.makedirs(self.vector_db_dir, exist_ok=True)
        
        self.ingested_files = self._load_ingested_files()
    
    def _load_ingested_files(self) -> List[str]:
        if os.path.exists(INGESTED_FILES_RECORD):
            with open(INGESTED_FILES_RECORD, "r") as f:
                return json.load(f)
        return []
    
    def _save_ingested_files(self, file_list: List[str]) -> None:
        with open(INGESTED_FILES_RECORD, "w") as f:
            json.dump(file_list, f)
    
    def _get_new_files(self) -> List[str]:
        all_files = get_document_files(self.documents_dir)
        pdf_files = [f for f in all_files if f.endswith(".pdf")]
        new_files = [f for f in pdf_files if f not in self.ingested_files]
        return new_files
    
    def load_document(self, file_path: str) -> List[Any]:
        try:
            loader = PyPDFLoader(file_path)
            return loader.load()
        except Exception as e:
            logger.error(f"Erreur lors du chargement du document {file_path}: {str(e)}")
            return []
    
    def process_documents(self, new_files: List[str]) -> List[Any]:
        all_documents = []
        
        for file_path in new_files:
            logger.info(f"Chargement du document: {file_path}")
            documents = self.load_document(file_path)
            all_documents.extend(documents)
        
        if not all_documents:
            logger.warning("Aucun contenu extrait des nouveaux documents.")
            return []
        
        chunks = self.text_splitter.split_documents(all_documents)
        logger.info(f"{len(chunks)} chunks créés à partir de {len(new_files)} documents.")
        return chunks
    
    def create_vector_store(self, chunks: List[Any]) -> Optional[Chroma]:
        if not chunks:
            logger.error("Aucun chunk à indexer.")
            return None
        
        try:
            if os.path.exists(self.vector_db_dir) and os.listdir(self.vector_db_dir):
                logger.info("Base vectorielle existante trouvée. Ajout des nouveaux chunks.")
                vectorstore = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=self.vector_db_dir
                )
                vectorstore.add_documents(chunks)
            else:
                logger.info("Création d'une nouvelle base vectorielle.")
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory=self.vector_db_dir
                )
            
            vectorstore.persist()
            logger.info("Base vectorielle mise à jour avec succès.")
            return vectorstore
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour de la base vectorielle: {str(e)}")
            return None
    
    def run_ingestion(self) -> bool:
        logger.info("🔍 Vérification des nouveaux documents PDF à ingérer...")
        new_files = self._get_new_files()
        
        if not new_files:
            logger.info("✅ Aucun nouveau document PDF trouvé. Rien à faire.")
            return False
        
        logger.info(f"📄 {len(new_files)} nouveau(x) document(s) détecté(s) à ingérer.")
        chunks = self.process_documents(new_files)
        
        if not chunks:
            return False
        
        vectorstore = self.create_vector_store(chunks)
        if vectorstore:
            self.ingested_files.extend(new_files)
            self._save_ingested_files(self.ingested_files)
            logger.info("✅ Ingestion terminée avec succès.")
            return True
        return False

if __name__ == "__main__":
    ingestion = DocumentIngestion()
    success = ingestion.run_ingestion()
    if not success:
        logger.info("🚫 Fin du processus : aucun document n'a été ingéré.")
