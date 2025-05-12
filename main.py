from fastapi import FastAPI
from schemas.request_response import QueryRequest, QueryResponse
from services.answer_service import get_final_answer

app = FastAPI()

@app.post("/ask", response_model=QueryResponse)
async def ask_query(req: QueryRequest):
    return get_final_answer(req.query)