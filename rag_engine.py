from typing import Optional
import os
import shutil

# LangChain imports (versions compatibles)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

# Imports locaux
from config import VECTOR_DB_DIR, EMBEDDING_MODEL, GEMINI_API_KEY, GEMINI_MODEL, SYSTEM_PROMPT
from utils import logger

class RAGEngine:
    def __init__(self, reset_db=False):
        logger.info("Initialisation du moteur RAG...")
        try:
            if reset_db and os.path.exists(VECTOR_DB_DIR):
                logger.info(f"Réinitialisation de la base vectorielle dans {VECTOR_DB_DIR}")
                shutil.rmtree(VECTOR_DB_DIR)
                os.makedirs(VECTOR_DB_DIR, exist_ok=True)
                logger.info("Base vectorielle réinitialisée")

            if not os.path.exists(VECTOR_DB_DIR):
                os.makedirs(VECTOR_DB_DIR, exist_ok=True)
                logger.info(f"Création du répertoire {VECTOR_DB_DIR}")

            if not os.listdir(VECTOR_DB_DIR):
                logger.warning("Base vectorielle vide - le système nécessitera une ingestion de documents")

            self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

            try:
                self.vectorstore = Chroma(
                    persist_directory=VECTOR_DB_DIR,
                    embedding_function=self.embeddings
                )
                collection_count = len(self.vectorstore._collection.get()["ids"])
                logger.info(f"Base vectorielle chargée avec {collection_count} documents")
            except Exception as e:
                logger.warning(f"Erreur lors du chargement de la base existante: {str(e)}")
                logger.info("Tentative de création d'une nouvelle collection...")
                if os.path.exists(VECTOR_DB_DIR):
                    shutil.rmtree(VECTOR_DB_DIR)
                    os.makedirs(VECTOR_DB_DIR, exist_ok=True)

                self.vectorstore = Chroma(
                    persist_directory=VECTOR_DB_DIR,
                    embedding_function=self.embeddings
                )
                logger.warning("Une nouvelle base vectorielle vide a été créée, veuillez ingérer des documents")

            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )

            self.llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key=GEMINI_API_KEY,
                temperature=0.0,
                top_p=0.95,
                top_k=40,
                max_output_tokens=2048,
            )

            template = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human",
                 """Contexte: {context}

                 Question: {question}

                 Réponds uniquement avec les informations fournies dans le contexte.
                 Si tu ne trouves pas l'information, dis simplement que tu ne sais pas 
                 et suggère de contacter un conseiller Baobab.""")
            ])

            self.chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | template
                | self.llm
                | StrOutputParser()
            )

            logger.info("✅ Moteur RAG initialisé avec succès")

        except Exception as e:
            logger.error(f"Erreur init moteur RAG: {str(e)}")
            raise

    def ask(self, query: str) -> str:
        try:
            logger.info(f"🔍 Question posée: {query}")
            result = self.chain.invoke(query)
            logger.info("✅ Réponse générée avec succès")
            return result
        except Exception as e:
            logger.error(f"Erreur génération réponse: {e}")
            raise


def get_rag_engine(reset_db=False) -> Optional[RAGEngine]:
    try:
        return RAGEngine(reset_db=reset_db)
    except Exception as e:
        logger.error(f"Erreur création moteur RAG: {str(e)}")
        return None
