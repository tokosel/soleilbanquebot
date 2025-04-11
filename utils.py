import logging

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('soleil_bank_bot')

def validate_query(query):
    """Valide que la requête est suffisamment longue et non vide."""
    return query and len(query.strip()) >= 3

def sanitize_input(text):
    """Nettoie l'entrée utilisateur pour éviter les injections."""
    # Exemple basique - à renforcer selon les besoins
    return text.strip()

def error_response(message):
    """Crée une réponse d'erreur standardisée."""
    return {"error": True, "message": message}