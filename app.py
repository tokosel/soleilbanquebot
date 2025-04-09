from flask import Flask, render_template, request, jsonify
import os
import threading
import json

# Imports locaux
from config import FLASK_SECRET_KEY, FLASK_DEBUG, FLASK_HOST, FLASK_PORT, DOCUMENTS_DIR
from utils import logger, validate_query, sanitize_input, error_response
from rag_engine import get_rag_engine

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

# Configuration Flask pour gérer les caractères spéciaux
app.config['JSON_AS_ASCII'] = False

# Singleton pour le moteur RAG
rag_engine = None
rag_engine_lock = threading.Lock()

def get_or_initialize_rag():
    global rag_engine
    with rag_engine_lock:
        if rag_engine is None:
            logger.info("Initialisation du moteur RAG")
            rag_engine = get_rag_engine()
    return rag_engine

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
        
        engine = get_or_initialize_rag()
        if engine is None:
            return jsonify(error_response("Système non disponible, veuillez réessayer plus tard")), 503
        
        # Récupération de la réponse sous forme de chaîne
        response = engine.ask(query)
        
        # Vérification que la réponse est bien une chaîne de caractères
        if not isinstance(response, str):
            response = str(response)
            
        # Log de débogage pour voir la réponse avant de l'envoyer
        logger.debug(f"Réponse avant envoi (50 premiers caractères): {response[:50]}...")
        
        # Test de sérialisation pour détecter les problèmes potentiels
        try:
            test_json = json.dumps({"answer": response})
            logger.debug("Sérialisation JSON réussie")
        except Exception as json_error:
            logger.error(f"Erreur de sérialisation JSON: {str(json_error)}")
            response = "Erreur de formatage dans la réponse. Veuillez reformuler votre question."
        
        return jsonify({"answer": response})
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la requête: {str(e)}")
        return jsonify(error_response("Une erreur est survenue")), 500

@app.route('/api/status')
def status():
    engine = get_or_initialize_rag()
    return jsonify({
        "status": "online" if engine is not None else "initializing",
        "ready": engine is not None
    })

if __name__ == '__main__':
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)