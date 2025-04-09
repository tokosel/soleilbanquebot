from typing import Optional
import os
import shutil

# LangChain imports (versions compatibles)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
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

            # Création du template de prompt avec messages
            self.template = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human",
                 """Contexte: {context}

                 Question: {question}

                 Réponds uniquement avec les informations fournies dans le contexte.
                 Si tu ne trouves pas l'information, dis simplement que tu ne sais pas 
                 et suggère de contacter un conseiller Soleil.""")
            ])

            logger.info("✅ Moteur RAG initialisé avec succès")

        except Exception as e:
            logger.error(f"Erreur init moteur RAG: {str(e)}")
            raise

    def ask(self, query: str) -> str:
        try:
            logger.info(f"🔍 Question posée: {query}")

            # Utiliser invoke() au lieu de get_relevant_documents()
            docs = self.retriever.invoke(query)
            context = " ".join([doc.page_content for doc in docs]) if docs else ""

            # Utilisation de format_messages pour obtenir une liste structurée de messages
            messages = self.template.format_messages(context=context, question=query)
            
            # Appel direct du LLM via invoke avec les messages structurés
            response = self.llm.invoke(messages)
            
            # Extraction du contenu texte de l'AIMessage
            if hasattr(response, 'content'):
                answer = response.content
            elif isinstance(response, str):
                answer = response
            else:
                # Fallback pour d'autres types de réponses
                answer = str(response)
                
            # Log pour déboguer le type et contenu de la réponse
            logger.info(f"Type de réponse: {type(answer)}")
            logger.debug(f"Contenu de la réponse: {answer[:50]}...")

            logger.info("✅ Réponse générée avec succès")
            return answer

        except Exception as e:
            logger.error(f"Erreur génération réponse: {e}")
            return f"Une erreur est survenue lors du traitement de votre demande: {str(e)}"


def get_rag_engine(reset_db=False) -> Optional[RAGEngine]:
    try:
        return RAGEngine(reset_db=reset_db)
    except Exception as e:
        logger.error(f"Erreur création moteur RAG: {str(e)}")
        return None