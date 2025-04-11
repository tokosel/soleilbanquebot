import chromadb

class Indexer:
    def __init__(self, db_path):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection("soleil-banque")

    def index_chunks(self, chunks, doc_name):
        """Ajoute les chunks à la base vectorielle."""
        for i, chunk in enumerate(chunks):
            self.collection.add(
                ids=[f"{doc_name}_{i}"],
                documents=[chunk]
            )
