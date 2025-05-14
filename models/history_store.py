# qa_history = []
# qa_prompt = []   # 프롬프트를 저장

# def add_qa(question, answer):
#     qa_history.append({"question": question, "answer": answer})
#     qa_prompt.append(f"[질문] {question}\n[답변] {answer}\n")

# def get_history():
#     return qa_history

# def get_history_string():
#     return "\n".join(qa_prompt)


from langchain_core.chat_history import BaseChatMessageHistory

class InMemoryHistory(BaseChatMessageHistory):
    def __init__(self):
        self.messages = []

    def add_messages(self, messages):
        self.messages.extend(messages)
        
    def clear(self):
        self.messages = []

    def __repr__(self):
        return str(self.messages)


def get_by_session_id(session_id):
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]

store = {}
