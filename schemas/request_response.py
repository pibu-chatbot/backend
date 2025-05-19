from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from models.history_store import get_by_session_id
from schemas.prompt_templates import chat_prompts


def get_chain(model):
   prompt = ChatPromptTemplate.from_messages([
      ("system", chat_prompts + '''
---
[검색된 문서]
{search_results}

[실제 질문]
{query}
'''),
      MessagesPlaceholder(variable_name="history"),
      ("human", "{query}")
   ])
   chain = prompt | model
   return RunnableWithMessageHistory(
      chain,
      get_session_history=get_by_session_id,
      input_messages_key="query",
      history_messages_key="history"
   )
