import pandas as pd
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

ingredients_df = pd.read_csv("preprocessed_ingredients.csv")
cosmetics_df = pd.read_csv("oliveyoung_final.csv")
print('csv 변환')

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print('임베딩 모델 완료')

documents = []
cosmetics = []

# 성분 Document
for _, row in ingredients_df.iterrows():
    content = row['description']
    metadata = {
        "type": "ingredient",
        "name": row['ingredient']
    }
    documents.append(Document(page_content=content, metadata=metadata))
    
# 화장품 Document
for _, row in cosmetics_df.iterrows():
    content = f"제품명: {row['product_name']}, 성분: {row['ingredient']}, 리뷰: {row['reviews']}, 사용법: {row['usage']}"
    metadata = {
        "type": "cosmetic",
        "product_name": row['product_name']
    }
    documents.append(Document(page_content=content, metadata=metadata))
print('document 생성')

vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

vectordb.persist()
print("ChromaDB에 저장 완료")