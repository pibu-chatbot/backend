from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from models.qlora_model import load_qlora_llm

# 1. 임베딩 모델 준비
embedding_model = HuggingFaceEmbeddings(
    model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS"
)

# 2. 화장품 ChromaDB 인스턴스 준비 (일단 cosmetic만)
cosmetic_chroma = Chroma(
    persist_directory="./cosmetic_chromadb",
    embedding_function=embedding_model
)

# 3. 리트리버 생성 (cosmetic만!)
cosmetic_retriever = cosmetic_chroma.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.0, "k": 2}
)

# 4. 질문 입력
user_question = "건성 피부가 겨울에 쓰기 좋은 로션 추천해줘"

# 5. 배경지식 RAG로 검색
docs = cosmetic_retriever.invoke(user_question)
background_knowledge = "\n".join([doc.page_content for doc in docs])

# 6. QLoRA LLM 로드
llm = load_qlora_llm(
    base_model_path="beomi/KoAlpaca-Polyglot-5.8B",
    adapter_path="./qlora_model_koalpaca"
)

# 7. 프롬프트 생성 (파인튜닝 포맷과 일치)
prompt = f"""질문: {user_question}
배경 지식: {background_knowledge}
답변:"""

# 8. LangChain LLM 인터페이스에 맞게 메시지 리스트로 전달
from langchain_core.messages import HumanMessage
result = llm.invoke([HumanMessage(content=prompt)])
print(result.content)
