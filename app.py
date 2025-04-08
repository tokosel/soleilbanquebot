from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import threading

# Imports locaux
from config import FLASK_SECRET_KEY, FLASK_DEBUG, FLASK_HOST, FLASK_PORT, DOCUMENTS_DIR
from utils import logger, validate_query, sanitize_input, format_response, error_response
from rag_engine import get_rag_engine
from ingestion import add_document, DocumentIngestion

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
        # Récupération et validation de la question
        data = request.json
        if not data or 'query' not in data:
            return jsonify(error_response("Question manquante")), 400
            
        query = sanitize_input(data['query'])
        if not validate_query(query):
            return jsonify(error_response("Question trop courte ou invalide")), 400
            
        # Récupération du moteur RAG
        engine = get_or_initialize_rag()
        if engine is None:
            return jsonify(error_response("Système non disponible, veuillez réessayer plus tard")), 503
            
        # Génération de la réponse
        response = engine.ask(query)
        
        return jsonify(format_response(response))
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la requête: {str(e)}")
        return jsonify(error_response("Une erreur est survenue")), 500

@app.route('/api/upload', methods=['POST'])
def upload_document():
    """Endpoint pour télécharger un document"""
    try:
        if 'file' not in request.files:
            return jsonify(error_response("Aucun fichier n'a été envoyé")), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify(error_response("Nom de fichier invalide")), 400
            
        # Vérification de l'extension
        filename = secure_filename(file.filename)
        _, ext = os.path.splitext(filename)
        if ext.lower() not in ['.pdf', '.docx', '.txt', '.html']:
            return jsonify(error_response("Format de fichier non supporté")), 400
            
        # Sauvegarde temporaire du fichier
        temp_path = os.path.join(DOCUMENTS_DIR, filename)
        file.save(temp_path)
        
        # Lancement de l'ingestion en arrière-plan
        def process_document():
            try:
                ingestion = DocumentIngestion()
                success = ingestion.run_ingestion()
                
                if success:
                    logger.info(f"Document {filename} ingéré avec succès")
                    
                    # Réinitialisation du moteur RAG
                    global rag_engine
                    with rag_engine_lock:
                        rag_engine = None
                else:
                    logger.error(f"Échec de l'ingestion du document {filename}")
            except Exception as e:
                logger.error(f"Erreur pendant l'ingestion: {str(e)}")
        
        # Lancement du thread d'ingestion
        thread = threading.Thread(target=process_document)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "success": True,
            "message": f"Document {filename} téléchargé et en cours de traitement"
        })
        
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement: {str(e)}")
        return jsonify(error_response("Une erreur est survenue lors du téléchargement")), 500

@app.route('/api/status')
def status():
    """Endpoint pour vérifier le statut du chatbot"""
    engine = get_or_initialize_rag()
    return jsonify({
        "status": "online" if engine is not None else "initializing",
        "ready": engine is not None
    })

if __name__ == '__main__':
    # S'assurer que les répertoires nécessaires existent
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    
    # Démarrage de l'application Flask
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)