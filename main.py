from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from models.embedder import embed_text
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str

### gpt ###
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

from schemas import request_response

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(model_name='gpt-4o-mini', temperature=0.4)

### q-lora ###
# from models. qlora_model import LocalHuggingFaceChat, model, tokenizer
# q_lora_model= LocalHuggingFaceChat(model=model, tokenizer=tokenizer)

# model 넘기기
chain_with_history = request_response.get_chain_with_model(model)#q_lora_model

app = FastAPI()

# CORS 설정 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask")
def ask_question(request: QueryRequest):
    # answer = embed_text(request.query)
    response = chain_with_history.invoke({"query": request.query}, config={"configurable": {"session_id": "test"}})
    print(response)
    return {"answer": response.content}

# uvicorn main:app --reload