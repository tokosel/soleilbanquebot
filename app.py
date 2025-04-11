from flask import Flask, render_template, request, jsonify
import os
import threading

# Imports locaux
from retriever import Retriever
from model_config import Model
from utils import logger, validate_query, sanitize_input, error_response

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "soleil_bank_default_key")

# Configuration Flask pour gérer les caractères spéciaux
app.config['JSON_AS_ASCII'] = False

# Singleton pour le retriever
retriever = None
retriever_lock = threading.Lock()

def get_or_initialize_retriever():
    global retriever
    with retriever_lock:
        if retriever is None:
            logger.info("Initialisation du système de récupération de documents")
            retriever = Retriever()
    return retriever

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify(error_response("Question manquante")), 400
        
        query = sanitize_input(data['query'])
        if not validate_query(query):
            return jsonify(error_response("Question trop courte ou invalide")), 400
        
        # Récupération du retriever
        r = get_or_initialize_retriever()
        if r is None:
            return jsonify(error_response("Système non disponible, veuillez réessayer plus tard")), 503
        
        # Récupération des documents pertinents
        documents = r.retrieve_documents(query)
        context = "\n".join(documents) if documents else ""
        
        # Génération de la réponse
        response = Model.generate_response(context, query)
        
        return jsonify({"answer": response})
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la requête: {str(e)}")
        return jsonify(error_response("Une erreur est survenue")), 500

@app.route('/api/status')
def status():
    r = get_or_initialize_retriever()
    return jsonify({
        "status": "online" if r is not None else "initializing",
        "ready": r is not None
    })

if __name__ == '__main__':
    # Définition des variables d'environnement par défaut si non spécifiées
    FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))
    FLASK_DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    # Message de démarrage
    logger.info(f"Démarrage de l'application Soleil Chatbot sur {FLASK_HOST}:{FLASK_PORT}, mode debug: {FLASK_DEBUG}")
    
    # Lancement de l'application
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
