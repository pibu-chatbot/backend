# 질문 임베딩

import torch
from sentence_transformers import SentenceTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

def embed_text(text):
    return model.encode(text, convert_to_numpy=True)