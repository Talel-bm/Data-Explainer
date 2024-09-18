from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from src.data_loader import load_csv_directory
from src.document_processor import load_metadata, enhance_documents_with_metadata
from config import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME

def setup_vector_store(csv_folder):
    csv_documents = load_csv_directory(csv_folder)
    metadata = load_metadata(csv_folder)
    enhanced_documents = enhance_documents_with_metadata(csv_documents, metadata)

    embed_model = HuggingFaceEmbedding()

    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = db.get_or_create_collection(CHROMA_COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        enhanced_documents,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    return index
