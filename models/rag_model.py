# RAG 기반 질의 응답

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# def embed_texts(texts):
#     return model.encode(texts, convert_to_numpy=True)

def search_documents(text):
    print('search_documents: ', text)
    vectordb = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embedding_model
    )
    # vector_text = embed_texts([text])[0]

    results = vectordb.similarity_search_with_score(text, k=3)
    print('results: ', results)
    return results

# if __name__=='__main__':
#     search_documents('여드름 피부에 좋은 성분은?')