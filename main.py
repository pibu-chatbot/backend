import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI

from models.embedder import embed_text
from schemas.request_response import get_chain


class QueryRequest(BaseModel):
    query: str

def load_openai_api_key():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_api_key

def set_cors(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

def build_model():
    model = ChatOpenAI(
        model_name='gpt-4o-mini', 
        temperature=0.4, 
        token=load_openai_api_key()
    )
    return model

def create_app():
    app = FastAPI() 
    load_openai_api_key()
    set_cors(app)
    return app

app = create_app()

if __name__=='__main__':
    
    model = build_model()
    chat_memory = get_chain(model)



@app.post("/ask")
def ask_question(request: QueryRequest):
    # answer = embed_text(request.query)
    response = chain_with_history.invoke({"query": request.query}, config={"configurable": {"session_id": "test"}})
    print(response)
    return {"answer": response.content}

# uvicorn main:app --reload