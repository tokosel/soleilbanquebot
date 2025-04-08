import os
import shutil
from typing import List, Dict, Any, Optional
import argparse

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
        """
        Initialise le processus d'ingestion de documents
        
        Args:
            documents_dir: Répertoire où sont stockés les documents
            vector_db_dir: Répertoire où sera stockée la base vectorielle
            chunk_size: Taille des chunks de texte
            chunk_overlap: Chevauchement entre les chunks
            embedding_model: Modèle d'embedding à utiliser
        """
        self.documents_dir = documents_dir
        self.vector_db_dir = vector_db_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialisation du text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialisation du modèle d'embedding
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # S'assurer que les répertoires existent
        os.makedirs(self.documents_dir, exist_ok=True)
        os.makedirs(self.vector_db_dir, exist_ok=True)
        
    def load_document(self, file_path: str) -> List[Any]:
        """
        Charge un document en fonction de son extension
        
        Args:
            file_path: Chemin vers le fichier à charger
            
        Returns:
            Une liste de documents chargés
        """
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
        """
        Traite tous les documents dans le répertoire et les découpe en chunks
        
        Returns:
            Une liste de chunks de documents
        """
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
        
        # Découpage des documents en chunks
        chunks = self.text_splitter.split_documents(all_documents)
        logger.info(f"Nombre total de chunks créés: {len(chunks)}")
        
        return chunks
    
    def create_vector_store(self, chunks: List[Any]) -> Optional[Chroma]:
        """
        Crée une base vectorielle à partir des chunks de documents
        
        Args:
            chunks: Liste de chunks de documents
            
        Returns:
            Une instance de ChromaDB ou None en cas d'erreur
        """
        if not chunks:
            logger.error("Aucun chunk à indexer")
            return None
            
        try:
            # Supprimer l'ancienne base si elle existe
            if os.path.exists(self.vector_db_dir) and os.listdir(self.vector_db_dir):
                logger.info("Nettoyage de l'ancienne base vectorielle")
                shutil.rmtree(self.vector_db_dir)
                os.makedirs(self.vector_db_dir)
                
            # Création de la nouvelle base vectorielle
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.vector_db_dir
            )
            
            # Persistance de la base
            vectorstore.persist()
            logger.info(f"Base vectorielle créée avec succès dans {self.vector_db_dir}")
            
            return vectorstore
        except Exception as e:
            logger.error(f"Erreur lors de la création de la base vectorielle: {str(e)}")
            return None
    
    def run_ingestion(self) -> bool:
        """
        Exécute le processus d'ingestion complet
        
        Returns:
            True si l'ingestion a réussi, False sinon
        """
        logger.info("Démarrage du processus d'ingestion")
        
        # Traitement des documents
        chunks = self.process_documents()
        if not chunks:
            return False
            
        # Création de la base vectorielle
        vectorstore = self.create_vector_store(chunks)
        
        return vectorstore is not None

def add_document(file_path: str, destination_dir: str = DOCUMENTS_DIR) -> bool:
    """
    Ajoute un document au répertoire des documents
    
    Args:
        file_path: Chemin vers le fichier à ajouter
        destination_dir: Répertoire de destination
        
    Returns:
        True si l'ajout a réussi, False sinon
    """
    try:
        if not os.path.isfile(file_path):
            logger.error(f"Le fichier {file_path} n'existe pas")
            return False
            
        extension = extract_file_extension(file_path)
        if extension not in ['.pdf', '.docx', '.txt', '.html']:
            logger.error(f"Extension non supportée: {extension}")
            return False
            
        # Copie du fichier dans le répertoire des documents
        shutil.copy2(file_path, destination_dir)
        logger.info(f"Document {file_path} ajouté avec succès")
        
        return True
    except Exception as e:
        logger.error(f"Erreur lors de l'ajout du document: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestion de documents pour le chatbot Baobab")
    parser.add_argument("--add", type=str, help="Ajouter un document au répertoire des documents")
    parser.add_argument("--ingest", action="store_true", help="Lancer le processus d'ingestion")
    
    args = parser.parse_args()
    
    if args.add:
        success = add_document(args.add)
        if not success:
            exit(1)
            
    if args.ingest or args.add:
        ingestion = DocumentIngestion()
        success = ingestion.run_ingestion()
        if not success:
            exit(1)
    else:
        parser.print_help()