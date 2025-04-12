import os
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

# Chemins de base
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = os.path.join(BASE_DIR, "data")
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
VECTOR_DB_DIR = os.path.join(DATA_DIR, "vector_store")
# Configuration de l'API Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class Model:
    @staticmethod
    def generate_response(context, query):
        """Utilise Gemini pour générer une réponse basée sur le contexte et la question."""
        
        prompt = f"""Tu es un assistant virtuel bancaire travaillant pour la Banque Soleil.
                    Ton rôle est d'aider les clients avec leurs questions sur les produits et services bancaires.

                    Utilise uniquement les informations suivantes pour répondre à la question du client.
                    Si tu ne trouves pas l'information dans les passages fournis, indique poliment que 
                    tu ne disposes pas de cette information et suggère de contacter un la Banque Soleil
                    Tél : (+221) 33 839 55 00
                    Email : contact@banquesoleil.sn
                    Site web : www.banquesoleil.sn.

        Contexte : {context}

        Question du client : {query}

        Réponse :"""

        model = genai.GenerativeModel("gemini-1.5-flash-8b-exp-0827")
        response = model.generate_content(prompt)
        return response.text
