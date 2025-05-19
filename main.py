import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI

from models.embedder import embed_text
from schemas.request_response import get_chain
from models.rag_model import search_documents


class QueryRequest(BaseModel):
    query: str
    session_id: str

def load_api_key():
    load_dotenv()

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
        temperature=0.4
    )
    return model

def create_app():
    app = FastAPI() 
    set_cors(app)
    return app

app = create_app()
load_api_key()
model = build_model()
chat_memory = get_chain(model)

@app.post("/ask")
def ask_question(request: QueryRequest):
    search_results = search_documents(request.query)
    print('search_results:', search_results)
    response = chat_memory.invoke(
        {"query": request.query, "search_results": search_results}, 
        config={"configurable": {"session_id": ""}}
    )
    print(response)
    return {"answer": response.content}

if __name__=='__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    
