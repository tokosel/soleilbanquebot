import os
import logging
from typing import List, Dict, Any, Optional

def setup_logger():
    """Configure un logger basique pour le projet"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('baobab_chatbot')

logger = setup_logger()

def extract_file_extension(file_path: str) -> str:
    """Extrait l'extension d'un fichier"""
    _, extension = os.path.splitext(file_path)
    return extension.lower()

def get_document_files(directory: str) -> List[str]:
    """Récupère tous les fichiers de documents dans un répertoire"""
    valid_extensions = ['.pdf', '.docx', '.txt', '.html']
    files = []
    
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path) and extract_file_extension(file_path) in valid_extensions:
            files.append(file_path)
            
    return files

def validate_query(query: str) -> bool:
    """Vérifie si une requête est valide (non vide et contient au moins 3 caractères)"""
    if not query or len(query.strip()) < 3:
        return False
    return True

def sanitize_input(text: str) -> str:
    """Nettoie les entrées utilisateur pour éviter les injections"""
    # Implémentation de base, à améliorer en production
    return text.strip()

def format_response(response_data: Dict[Any, Any]) -> Dict[str, Any]:
    """Formate la réponse du chatbot de manière cohérente"""
    formatted_response = {
        "answer": response_data.get("answer", ""),
        "sources": response_data.get("sources", []),
        "success": True,
        "error": None
    }
    
    return formatted_response

def error_response(message: str) -> Dict[str, Any]:
    """Crée une réponse d'erreur formatée"""
    return {
        "answer": "",
        "sources": [],
        "success": False,
        "error": message
    }

def mask_sensitive_info(text: str) -> str:
    """Masque les informations sensibles comme les numéros de compte"""
    # Implémentation basique - à enrichir selon les besoins spécifiques
    import re
    
    # Masquage des numéros qui ressemblent à des comptes bancaires
    # (cette regex est simpliste et doit être adaptée aux formats spécifiques de Baobab)
    text = re.sub(r'\b\d{10,16}\b', '************', text)
    
    return text