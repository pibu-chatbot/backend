from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from langchain_core.messages import HumanMessage
from request_response import get_chain_with_model
import torch
import os
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
openai_token = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 및 토크나이저 로드
base_model_path = "mistralai/Mistral-7B-Instruct-v0.1"
adapter_path = "./qlora_model_mistral"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    use_auth_token=hf_token
)

# 2. adapter 병합
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload() 

# 3. tokenizer도 base 모델 기준
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# LangChain용 chain 생성
chain_with_history = get_chain_with_model(model)

# 입력 스키마 정의
class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"

# API 엔드포인트
@app.post("/ask")
def ask_question(request: QueryRequest):
    result = chain_with_history.invoke(
        {"query": request.query},
        config={"configurable": {"session_id": request.session_id}}
    )
    return {"answer": result.content}
