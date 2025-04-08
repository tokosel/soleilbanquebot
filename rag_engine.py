from typing import Dict, List, Any, Optional
import os

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

# Imports locaux
from config import VECTOR_DB_DIR, EMBEDDING_MODEL, GEMINI_API_KEY, GEMINI_MODEL, SYSTEM_PROMPT
from utils import logger, mask_sensitive_info

class RAGEngine:
    def __init__(self, 
                 vector_db_dir: str = VECTOR_DB_DIR,
                 embedding_model: str = EMBEDDING_MODEL,
                 gemini_api_key: str = GEMINI_API_KEY,
                 gemini_model: str = GEMINI_MODEL,
                 system_prompt: str = SYSTEM_PROMPT):
        """
        Initialise le moteur RAG avec LangChain et Gemini
        
        Args:
            vector_db_dir: Répertoire de la base vectorielle
            embedding_model: Modèle d'embedding à utiliser
            gemini_api_key: Clé API pour Gemini
            gemini_model: Modèle Gemini à utiliser
            system_prompt: Prompt système pour guider le modèle
        """
        self.vector_db_dir = vector_db_dir
        self.embedding_model = embedding_model
        self.gemini_api_key = gemini_api_key
        self.gemini_model = gemini_model
        self.system_prompt = system_prompt
        
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.chain = None
        
        # Initialisation du moteur
        self._initialize()
        
    def _initialize(self) -> None:
        """Initialise les composants du moteur RAG"""
        try:
            # Vérifier si la base vectorielle existe
            if not os.path.exists(self.vector_db_dir) or not os.listdir(self.vector_db_dir):
                logger.error(f"La base vectorielle n'existe pas dans {self.vector_db_dir}")
                raise ValueError("Base vectorielle manquante")
                
            # Initialisation des embeddings et de la base vectorielle
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
            
            self.vectorstore = Chroma(
                persist_directory=self.vector_db_dir,
                embedding_function=self.embeddings
            )
            
            # Création du retriever avec un filtre de similarité
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            # Initialisation du modèle Gemini
            self.llm = ChatGoogleGenerativeAI(
                model=self.gemini_model,
                google_api_key=self.gemini_api_key,
                temperature=0.0,
                top_p=0.95,
                top_k=40,
                max_output_tokens=2048,
            )
            
            # Template pour le prompt
            template = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("human", 
                 """Contexte: {context}
                 
                 Question: {question}
                 
                 Réponds uniquement avec les informations fournies dans le contexte. 
                 Si tu ne trouves pas l'information, dis simplement que tu ne sais pas 
                 et suggère de contacter un conseiller Baobab.""")
            ])
            
            # Construction de la chaîne RAG
            self.chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | template
                | self.llm
                | StrOutputParser()
            )
            
            logger.info("Moteur RAG initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du moteur RAG: {str(e)}")
            raise
    
    def ask(self, query: str) -> Dict[str, Any]:
        """
        Pose une question au chatbot
        
        Args:
            query: Question de l'utilisateur
            
        Returns:
            Un dictionnaire contenant la réponse et les sources
        """
        try:
            # Nettoyage de la requête
            query = mask_sensitive_info(query.strip())
            
            # Récupération des documents pertinents
            retrieved_docs = self.retriever.get_relevant_documents(query)
            
            sources = []
            for doc in retrieved_docs:
                if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                    source_path = doc.metadata['source']
                    source_name = os.path.basename(source_path)
                    if source_name not in sources:
                        sources.append(source_name)
            
            # Génération de la réponse
            answer = self.chain.invoke(query)
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de la réponse: {str(e)}")
            return {
                "answer": "Désolé, une erreur est survenue lors du traitement de votre question. Veuillez réessayer.",
                "sources": []
            }

def get_rag_engine() -> Optional[RAGEngine]:
    """
    Crée et retourne une instance du moteur RAG
    
    Returns:
        Une instance du moteur RAG ou None en cas d'erreur
    """
    try:
        return RAGEngine()
    except Exception as e:
        logger.error(f"Impossible de créer le moteur RAG: {str(e)}")
        return None