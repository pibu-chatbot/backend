from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.ensemble import EnsembleRetriever

def set_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS"
    )
    return embedding_model

def load_chromadbs(embedding_model):
    chroma1 = Chroma(
        persist_directory="./cosmetic_chromadb",
        embedding_function=embedding_model
    )
    chroma2 = Chroma(
        persist_directory="./ingredient_chromadb",
        embedding_function=embedding_model
    )
    return chroma1, chroma2

def create_ensemble_retriever(chroma1, chroma2):
    retriever1 = chroma1.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.0, "k": 3}
    )
    retriever2 = chroma2.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.0, "k": 3}
    )
    ensemble_retriever = EnsembleRetriever(retrievers=[retriever1, retriever2])
    return ensemble_retriever

def search_documents(ensemble_retriever, query: str, k: int = 3):
    docs = ensemble_retriever.invoke(query)
    return docs[:k]