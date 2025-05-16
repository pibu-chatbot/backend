# backend.py

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from typing import Any
import os
from dotenv import load_dotenv

from prompt import prompt_template

# 환경 변수 로드
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청 모델 정의
class Question(BaseModel):
    session_id: Any
    question: str

# 세션별 메모리 저장
memory_dict = {}

# 커스텀 프롬프트 템플릿
custom_prompt = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template=prompt_template
)

# Embedding, LLM, Retriever 구성
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
db = Chroma(persist_directory="./vectordb/chroma_db", embedding_function=embedding)
retriever = db.as_retriever(search_kwargs={"k": 3})

# 질문 처리 라우트
@app.post("/ask")
def ask_question(payload: Question):
    session_id = payload.session_id
    question = payload.question

    if session_id not in memory_dict:
        memory_dict[session_id] = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

    memory = memory_dict[session_id]

    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )

    result = rag_chain({"question": question})
    return {"answer": result["answer"]}
