import os
import shutil
from typing import List, Any, Optional

# LangChain et documents
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Imports locaux
from config import DOCUMENTS_DIR, VECTOR_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
from utils import logger, get_document_files, extract_file_extension

class DocumentIngestion:
    def __init__(self, 
                 documents_dir: str = DOCUMENTS_DIR, 
                 vector_db_dir: str = VECTOR_DB_DIR,
                 chunk_size: int = CHUNK_SIZE,
                 chunk_overlap: int = CHUNK_OVERLAP,
                 embedding_model: str = EMBEDDING_MODEL):
        self.documents_dir = documents_dir
        self.vector_db_dir = vector_db_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        os.makedirs(self.documents_dir, exist_ok=True)
        os.makedirs(self.vector_db_dir, exist_ok=True)
        
    def load_document(self, file_path: str) -> List[Any]:
        extension = extract_file_extension(file_path)
        
        try:
            if extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif extension == '.docx':
                loader = Docx2txtLoader(file_path)
            elif extension == '.txt':
                loader = TextLoader(file_path)
            elif extension == '.html':
                loader = UnstructuredHTMLLoader(file_path)
            else:
                logger.warning(f"Extension non supportée: {extension}")
                return []
                
            return loader.load()
        except Exception as e:
            logger.error(f"Erreur lors du chargement du document {file_path}: {str(e)}")
            return []
    
    def process_documents(self) -> List[Any]:
        all_documents = []
        file_paths = get_document_files(self.documents_dir)
        
        for file_path in file_paths:
            logger.info(f"Traitement du document: {file_path}")
            documents = self.load_document(file_path)
            all_documents.extend(documents)
            
        if not all_documents:
            logger.warning("Aucun document n'a été chargé")
            return []
            
        logger.info(f"Nombre total de documents chargés: {len(all_documents)}")
        
        chunks = self.text_splitter.split_documents(all_documents)
        logger.info(f"Nombre total de chunks créés: {len(chunks)}")
        
        return chunks
    
    def create_vector_store(self, chunks: List[Any]) -> Optional[Chroma]:
        if not chunks:
            logger.error("Aucun chunk à indexer")
            return None
            
        try:
            if os.path.exists(self.vector_db_dir) and os.listdir(self.vector_db_dir):
                logger.info("Nettoyage de l'ancienne base vectorielle")
                shutil.rmtree(self.vector_db_dir)
                os.makedirs(self.vector_db_dir)
                
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.vector_db_dir
            )
            
            vectorstore.persist()
            logger.info(f"Base vectorielle créée avec succès dans {self.vector_db_dir}")
            return vectorstore
        except Exception as e:
            logger.error(f"Erreur lors de la création de la base vectorielle: {str(e)}")
            return None
    
    def run_ingestion(self) -> bool:
        logger.info("Démarrage du processus d'ingestion")
        chunks = self.process_documents()
        if not chunks:
            return False
            
        vectorstore = self.create_vector_store(chunks)
        return vectorstore is not None

if __name__ == "__main__":
    ingestion = DocumentIngestion()
    success = ingestion.run_ingestion()
    if success:
        logger.info("Ingestion terminée avec succès.")
    else:
        logger.error("Échec de l’ingestion.")
