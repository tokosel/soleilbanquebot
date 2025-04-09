import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
# Chemins de base
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = os.path.join(BASE_DIR, "data")
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
VECTOR_DB_DIR = os.path.join(DATA_DIR, "vector_db")

# Configuration de l'application Flask
FLASK_SECRET_KEY = "baobab-chatbot-secret-key-change-in-production"
FLASK_DEBUG = True
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000
# Configuration de l'API Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-1.5-flash-8b-exp-0827"

# Configuration de ChromaDB
CHROMA_PERSIST_DIRECTORY = VECTOR_DB_DIR

# Configuration RAG
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Modèle pour sentence-transformers

# Prompt template pour le chatbot
SYSTEM_PROMPT = """Tu es un assistant virtuel pour la banque Soleil. 
Tu dois répondre aux questions des clients concernant les services bancaires, en te basant uniquement 
sur les informations fournies. Sois précis, professionnel et courtois.
Si tu ne connais pas la réponse, indique-le clairement et suggère de contacter un conseiller.
N'invente jamais d'informations sur les produits bancaires ou les procédures."""

# Création des répertoires s'ils n'existent pas
for directory in [DATA_DIR, DOCUMENTS_DIR, VECTOR_DB_DIR]:
    os.makedirs(directory, exist_ok=True)