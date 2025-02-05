import weaviate
from weaviate.classes.config import Configure
_weaviate_client = None

def create_client():
    global _weaviate_client
    _weaviate_client = weaviate.connect_to_local(host="weaviate")
    return _weaviate_client
        
def get_client():
    if _weaviate_client is None:
        raise RuntimeError("Weaviate client is not loaded!")
    return _weaviate_client
    
def create_collection(collection_name: str):
    client = get_client()
    if not client.collections.exists(collection_name):
        med_documents = client.collections.create(
            name = collection_name,
            vectorizer_config = Configure.Vectorizer.none(),
        )
    
def close_client_conection():
    client = get_client()
    client.close()