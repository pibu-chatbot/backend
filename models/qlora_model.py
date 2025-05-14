# QLoRA 모델 로드 및 후처리
# 1. 로드
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("./qlora_model_Bllossom/qlora_model_Bllossom",local_files_only=True)
model = AutoModelForCausalLM.from_pretrained("./qlora_model_Bllossom/qlora_model_Bllossom", torch_dtype=torch.float16,
    device_map="auto",local_files_only=True)


# 2. 후처리
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# LocalHuggingFaceChat은 LangChain의 기본적인 대화형 모델 구조를 따르며, 후처리, 메시지 포맷, 결과 반환을 포함하는 방식으로 모델을 통합하는 역할
class LocalHuggingFaceChat(BaseChatModel):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _convert_messages_to_prompt(self, messages) -> str:
        prompt = ""
        for message in messages:
            if isinstance(message, SystemMessage):
                prompt += f"[시스템] {message.content}\n"
            elif isinstance(message, HumanMessage):
                prompt += f"[사용자] {message.content}\n"
            elif isinstance(message, AIMessage):
                prompt += f"[AI] {message.content}\n"
        prompt += "[AI] "  # 마지막에 AI가 대답할 자리
        return prompt

    def invoke(self, input, config=None):
        prompt = self._convert_messages_to_prompt(input["messages"])

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=200)
        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # 마지막 AI 대답만 잘라내기 (원하는 방식으로 자르기)
        answer = decoded.split("[AI]")[-1].strip()

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=answer))]
        )
