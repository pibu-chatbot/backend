import pandas as pd
import ast
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

# 리스트 모양 문자열을 리스트로 변경
def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except Exception:
        return []
    
ingredients_df = pd.read_csv("preprocessed_ingredients.csv")
cosmetics_df = pd.read_csv("preprocessed_oliveyoung.csv")
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
    # 각 컬럼 안전하게 추출
    product_name = row.get('product_name', '')
    ingredient_names = row.get('ingredient', '')
    review_summaries = safe_literal_eval(row.get('review_summaries', '[]'))
    keywords = safe_literal_eval(row.get('keywords', '[]'))
    document_text = row.get('document', '')
    usage = row.get('usage', '')

    # page_content
    content = (
        f"제품명: {product_name}\n"
        f"성분: {ingredient_names}\n"
        f"핵심 리뷰: {', '.join(review_summaries)}\n"
        f"키워드: {', '.join(keywords)}\n"
        f"종합 리뷰: {document_text}\n"
        f"사용법: {usage}"
    )

    metadata = {
        "type": "cosmetic",
        "product_name": product_name,
    }

    cosmetic_document.append(
        Document(page_content=content, metadata=metadata)
    )

print('ingredient, cosmetic document 생성')

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