from flask import Flask, render_template, request, jsonify
import threading
import os

# Imports locaux
from config import FLASK_SECRET_KEY, FLASK_DEBUG, FLASK_HOST, FLASK_PORT
from utils import logger, validate_query, sanitize_input, format_response, error_response
from rag_engine import get_rag_engine

# Initialisation de l'application Flask
app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

# Singleton pour le moteur RAG
rag_engine = None
rag_engine_lock = threading.Lock()

def get_or_initialize_rag():
    """Récupère ou initialise le moteur RAG de manière thread-safe"""
    global rag_engine
    with rag_engine_lock:
        if rag_engine is None:
            logger.info("Initialisation du moteur RAG")
            rag_engine = get_rag_engine()
    return rag_engine

@app.route('/')
def index():
    """Page principale du chatbot"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Endpoint pour poser une question au chatbot"""
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

        response = engine.ask(query)
        return jsonify(format_response(response))

    except Exception as e:
        logger.error(f"Erreur lors du traitement de la requête: {str(e)}")
        return jsonify(error_response("Une erreur est survenue")), 500

@app.route('/api/status')
def status():
    """Endpoint pour vérifier le statut du chatbot"""
    engine = get_or_initialize_rag()
    return jsonify({
        "status": "online" if engine is not None else "initializing",
        "ready": engine is not None
    })

if __name__ == '__main__':
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
