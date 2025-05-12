# 질문 임베딩

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text: str):
    return model.encode(text, convert_to_numpy=True)

if __name__ == "__main__":
    user_input = "피부가 예민한데 보습도 잘되는 제품이 필요해요."
    embedding = embed_text(user_input)

    print("\n- 임베딩 벡터:")
    print(embedding)
    print(f"\n- 벡터 차원 수: {len(embedding)}")