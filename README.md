# Chatbot RAG pour Soleil Banque :)

Ce projet implémente un chatbot intelligent basé sur l'architecture RAG (Retrieval-Augmented Generation) pour la banque Soleil du Sénégal. Il permet aux clients d'obtenir des réponses précises à leurs questions concernant les services bancaires.

## Fonctionnalités

- Interface conversationnelle intuitive
- Traitement des questions liées à l'ouverture de compte, prêts, épargne, microcrédits, etc.
- Ingestion et indexation automatiques de documents (PDF, DOCX, TXT, HTML)
- Architecture RAG avec LangChain et Gemini
- Base vectorielle ChromaDB pour stocker et rechercher des informations
- API REST pour interagir avec le chatbot
- Interface utilisateur moderne avec Vue.js et TailwindCSS

## Structure du projet

```
baobab-chatbot/
│
├── app.py                  # Application Flask principale
├── ingestion.py            # Script d'ingestion des documents
├── rag_engine.py           # Logique RAG avec LangChain et Gemini
├── config.py               # Configuration du projet
├── utils.py                # Fonctions utilitaires
│
├── templates/              # Templates HTML pour Flask
│   └── index.html          # Interface utilisateur du chatbot
│
├── static/                 # Fichiers statiques
│   ├── css/                # Styles CSS
│   ├── js/                 # Scripts JavaScript
│   └── img/                # Images
│
├── data/                   # Dossier pour les données
│   ├── documents/          # Documents source à indexer
│   └── vector_db/          # Base de données vectorielle ChromaDB
│
├── requirements.txt        # Dépendances du projet
└── README.md               # Documentation du projet
```

## Prérequis

- Python 3.9+
- Clé API Gemini

## Installation

1. Cloner le dépôt :
   ```bash
   git clone https://github.com/votre-username/baobab-chatbot.git
   cd baobab-chatbot
   ```

2. Créer un environnement virtuel :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows : venv\Scripts\activate
   ```

3. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

4. Configurer la clé API Gemini :
   Modifiez `config.py` et ajoutez votre clé API Gemini :
   ```python
   GEMINI_API_KEY = "votre-clé-api-gemini"
   ```

## Utilisation

### Ingestion de documents

Pour indexer vos documents, placez-les dans le dossier `data/documents` puis exécutez :

```bash
python ingestion.py --ingest
```

Vous pouvez également ajouter un document spécifique :

```bash
python ingestion.py --add chemin/vers/document.pdf
```

### Démarrage du serveur

```bash
python app.py
```

Le chatbot sera accessible à l'adresse http://127.0.0.1:5000/

## Sécurité

Ce projet implémente les meilleures pratiques pour la sécurité des données bancaires :

- Masquage des informations sensibles (numéros de compte, etc.)
- Traitement local des données
- Nettoyage des entrées utilisateur
- Validation des fichiers téléchargés

À noter que pour un déploiement en production, des mesures supplémentaires sont nécessaires (authentification, chiffrement HTTPS, journalisation, etc.).

## Roadmap de déploiement

### Phase 1 : Déploiement local
- Installation sur un serveur interne
- Tests avec un ensemble limité d'utilisateurs
- Optimisation des performances

### Phase 2 : Déploiement cloud
- Migration vers une plateforme cloud (AWS, GCP, Azure ou Vercel)
- Mise en place d'un scaling automatique
- Intégration avec les systèmes bancaires existants
- Configuration des sauvegardes automatiques

## Contribuer

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou à soumettre une pull request.

## License

[MIT](LICENSE)