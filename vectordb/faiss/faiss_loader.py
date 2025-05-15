# FAISS 로드 및 유사 문서 검색 함수
from dotenv import load_dotenv
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# 0. 토큰
load_dotenv()

hf_token = os.getenv("HF_TOKEN")

# 1. csv 로드 및 사용할 llm 모델 로드
# cosmetics_df = pd.read_csv("cosmetics.csv", encoding='utf-8')
ingredients_df = pd.read_csv("../../data/preprocessed_ingredients.csv", encoding='utf-8')

model = SentenceTransformer("Bllossom/llama-3.2-Korean-Bllossom-3B")
if model.tokenizer.pad_token is None:
    model.tokenizer.pad_token = model.tokenizer.eos_token

print('✅ 1. csv 로드 및 사용할 llm 모델 로드')

# 2. 벡터화할 텍스트 추출
ingredient_texts = ingredients_df['description']
# product_texts = cosmetics_df.apply(
#     lambda row: f"{row['brand']} {row['product_name']} 성분: {row['ingredient']} 리뷰: {row['reviews']} 사용법: {row['usage']}", axis=1
# ).tolist()

all_texts = ingredient_texts
# all_texts = ingredient_texts + product_texts
all_embeddings = model.encode(all_texts, convert_to_numpy=True)

print('✅ 2. 벡터화할 텍스트 추출')

# 3. FAISS 인덱스 생성
embedding_dim = all_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(all_embeddings)

print('✅ 3. FAISS 인덱스 생성')

# 4. 메타데이터 저장
metadata = []

# 성분 설명 메타
for i, row in ingredients_df.iterrows():
    metadata.append({
        "type": "ingredient",
        "name": row['ingredient'],
        "description": row['description']
    })

# 화장품 정보 메타
# for i, row in cosmetics_df.iterrows():
#     metadata.append({
#         "type": "cosmetic",
#         "product_name": row['product_name'],
#         "brand": row['brand'],
#         "ingredients": row['ingredients'],
#         "reviews": row['review'],
#         "usage": row['usage'],

#     })

print('✅ 4. 메타데이터 저장')

# 5. 인덱스와 메타 데이터 저장
faiss.write_index(index, "cosmetic_ingredient.index")

with open("cosmetic_ingredient_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("✅ 5. 인덱스와 메타 데이터 저장")
