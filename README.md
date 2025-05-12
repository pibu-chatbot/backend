# backend
- 사용자 질문 벡터화하기
- Vector에서 유사한 문서 검색하기 (FAISS)
- RAG 모델에 질문 + 검색 결과 입력해서 초안 답변 생성
- QLoRA 모델로 정제
- 최종 답변 반환하기

<code>
backend/
├── main.py
├── models/
│   ├── rag_model.py        ← RAG 질의 응답 (LLM + 벡터DB)
│   ├── qlora_model.py      ← QLoRA 모델 로딩 및 inference
│   └── embedder.py         ← 질문 임베딩 (예: SentenceTransformer)
├── vectordb/
│   └── faiss_loader.py     ← FAISS 로드 및 유사 문서 검색 함수
└── services/
    └── answer_service.py   ← 전체 파이프라인 조합
</code>