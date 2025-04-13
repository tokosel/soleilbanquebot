import chromadb
import os
import uuid
from chromadb.errors import NotFoundError

class VectorStore:
    def __init__(self, db_path="data/vector_store/chroma_db"):
        """Initialise la base vectorielle pour le chatbot de la Banque Soleil."""
        # Création du répertoire si nécessaire
        os.makedirs(db_path, exist_ok=True)
        
        # Initialisation du client ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Récupération ou création de la collection "soleil-banque"
        try:
            self.collection = self.client.get_collection("soleil-banque")
        except NotFoundError:
            # La collection n'existe pas encore, la créer
            self.collection = self.client.create_collection("soleil-banque")
            print("[VectorStore] Collection 'soleil-banque' créée.")

    def search(self, query, k=10):
        """Recherche les documents les plus pertinents pour la requête."""
        results = self.collection.query(query_texts=[query], n_results=k)
        # Retourne le premier résultat (en supposant que results["documents"] est une liste de listes)
        if results["documents"] and results["documents"][0]:
            return results["documents"][0]
        return []

    def add_documents(self, documents, metadatas=None, ids=None):
        """Ajoute des documents à la base vectorielle."""
        if not ids:
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas if metadatas else [{}] * len(documents),
            ids=ids
        )
        return True
