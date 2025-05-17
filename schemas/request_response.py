# 요청 응답 정의

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from models.history_store import get_by_session_id

def get_chain(model):
   prompt = ChatPromptTemplate.from_messages([
      ('system', '당신은 화장품 전문가입니다.'), 
      MessagesPlaceholder(variable_name='history'),
      ('human', '{query}')
   ])
   chain = prompt | model
   return RunnableWithMessageHistory(
      chain,
      get_session_history=get_by_session_id,
      input_messages_key='query',
      history_messages_key='history'
   )

# rag랑할꺼면
#('system', '검색된 문서: {search_results}')을 human밑에 넣기

# 그리고 main.py에서 
# 맨 밑 ask_question 함수에

# search_results = search_documents(request.query) 넣고.. search_documents함수는 rag_model.py에 만들어야할듯? -> # 사용자의 질문 임베딩 -> Chroma에서 가장 유사한 문서 검색

# 무튼 그런다음에
    # # 문서와 함께 history 및 query 전달
    # response = chain_with_history.invoke(
    #     {"query": request.query, "search_results": search_results},
    #     config={"configurable": {"session_id": "test"}}
    # )
# 이렇게 넣기