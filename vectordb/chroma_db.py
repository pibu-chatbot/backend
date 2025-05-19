import pandas as pd
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

ingredients_df = pd.read_csv("preprocessed_ingredients.csv")
cosmetics_df = pd.read_csv("oliveyoung.csv")
print('csv 변환')

embedding_model = HuggingFaceEmbeddings(
    model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS"
)
print('임베딩 모델 완료')

ingredient_document = []
cosmetic_document = []

# 성분 Document
for _, row in ingredients_df.iterrows():
    content = row['description']
    metadata = {
        "type": "ingredient",
        "name": row['ingredient']
    }
    ingredient_document.append(Document(page_content=content, metadata=metadata))
    
# 화장품 Document
for _, row in cosmetics_df.iterrows():
    content = f"제품명: {row['product_name']}, 성분: {row['ingredient']}, 리뷰: {row['reviews']}, 사용법: {row['usage']}"
    metadata = {
        "type": "cosmetic",
        "product_name": row['product_name']
    }
    cosmetic_document.append(Document(page_content=content, metadata=metadata))
print('document 생성')

ingredient_vectordb = Chroma.from_documents(
    documents=ingredient_document,
    embedding=embedding_model,
    persist_directory="./ingredient_chromadb"
)

print("ingredient_chromadb 저장 완료")

cosmetic_vectordb = Chroma.from_documents(
    documents=cosmetic_document,
    embedding=embedding_model,
    persist_directory="./cosmetic_chromadb"
)

print("cosmetic_chromadb 저장 완료")