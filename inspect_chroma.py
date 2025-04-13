import chromadb
import argparse
import pprint

def connect_chroma(db_path):
    client = chromadb.PersistentClient(path=db_path)
    return client

def list_collections(client):
    collections = client.list_collections()
    print("📚 Collections disponibles :")
    for col in collections:
        print(f" - {col.name}")

def show_documents(collection, limit=10):
    data = collection.get()
    docs = data.get("documents", [])[:limit]
    metas = data.get("metadatas", [])[:limit]
    ids = data.get("ids", [])[:limit]

    print(f"📄 Affichage des {len(docs)} premiers documents :\n")
    for i, (doc, meta, id_) in enumerate(zip(docs, metas, ids), 1):
        print(f"--- Document {i} ---")
        print(f"ID : {id_}")
        print(f"Meta : {meta}")
        print(f"Contenu :\n{doc[:300]}...\n")  # Affiche les 300 premiers caractères

def search_query(collection, query, k=5):
    results = collection.query(query_texts=[query], n_results=k)
    docs = results.get("documents", [[]])[0]

    print(f"🔍 Top {k} résultats pour la requête : '{query}'\n")
    for i, doc in enumerate(docs, 1):
        print(f"--- Résultat {i} ---")
        print(doc[:300], "...\n")  # Affiche les 300 premiers caractères

def main():
    parser = argparse.ArgumentParser(description="Inspecteur de base ChromaDB")
    parser.add_argument("--db", type=str, default="data/vector_store/chroma_db", help="Chemin vers la base ChromaDB")
    parser.add_argument("--collection", type=str, default="soleil-banque", help="Nom de la collection à lire")
    parser.add_argument("--list", action="store_true", help="Lister les collections disponibles")
    parser.add_argument("--show", action="store_true", help="Afficher des documents")
    parser.add_argument("--search", type=str, help="Effectuer une recherche")
    parser.add_argument("--limit", type=int, default=5, help="Nombre max de documents à afficher")

    args = parser.parse_args()

    client = connect_chroma(args.db)

    if args.list:
        list_collections(client)
        return

    try:
        collection = client.get_collection(args.collection)
    except Exception as e:
        print(f"❌ Erreur : Collection '{args.collection}' introuvable.")
        return

    if args.show:
        show_documents(collection, limit=args.limit)

    if args.search:
        search_query(collection, args.search, k=args.limit)

if __name__ == "__main__":
    main()
