qa_history = []
qa_prompt = []   # 프롬프트를 저장

def add_qa(question, answer):
    qa_history.append({"question": question, "answer": answer})
    qa_prompt.append(f"[질문] {question}\n[답변] {answer}\n")

def get_history():
    return qa_history

def get_history_string():
    return "\n".join(qa_prompt)