# Chatbot RAG pour Soleil Banque :)

Ce projet implémente un chatbot intelligent basé sur l'architecture RAG (Retrieval-Augmented Generation) pour la banque Soleil du Sénégal. Il permet aux clients d'obtenir des réponses précises à leurs questions concernant les services bancaires.

## Fonctionnalités

- Interface conversationnelle intuitive
- Traitement des questions liées à l'ouverture de compte, prêts, épargne, microcrédits, etc.
- Ingestion et indexation automatiques de documents PDF
- Architecture RAG avec LangChain et Gemini
- Base vectorielle ChromaDB pour stocker et rechercher des informations
- API REST pour interagir avec le chatbot
- Interface utilisateur moderne avec Vue.js et TailwindCSS


## Installation

1. Cloner le dépôt :
   ```bash
   git clone https://github.com/tokosel/soleilbanquebot.git
   cd soleilbanquebot
   ```

2. Créer un environnement virtuel :
   ```bash
   python -m venv env
   venv\Scripts\activate
   ```

3. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

4. Configurer la clé API Gemini :
   Créer un fichier `.env` et ajoutez votre clé API Gemini :
   ```python
   GEMINI_API_KEY = "votre-clé-api-gemini"
   ```

## Utilisation

### Ingestion de documents

Pour indexer les documents dans le dossier `data/documents` il faut exécuter :

```bash
python pipeline.py
```

### Démarrage du serveur

```bash
python app.py
```

Le chatbot sera accessible à l'adresse http://127.0.0.1:5000/
