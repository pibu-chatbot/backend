# QLoRA 모델 로드 및 후처리
# 1. 로드
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# tokenizer = AutoTokenizer.from_pretrained("./qlora_model_Bllossom/qlora_model_Bllossom",local_files_only=True)
# model = AutoModelForCausalLM.from_pretrained("./qlora_model_Bllossom/qlora_model_Bllossom", torch_dtype=torch.float16,
#     device_map="auto",local_files_only=True)


# 2. 후처리
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from peft import PeftModel
# LocalHuggingFaceChat은 LangChain의 기본적인 대화형 모델 구조를 따르며, 후처리, 메시지 포맷, 결과 반환을 포함하는 방식으로 모델을 통합하는 역할
from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel

class LocalHuggingFaceChat(BaseChatModel):
    model: Any
    tokenizer: Any

    @property
    def _llm_type(self) -> str:
        return "local-huggingface-chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        prompt = self._convert_messages_to_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        # token_type_ids 제거
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=700)
        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print("모델 출력:", decoded)  # 추가

        answer = decoded.split("[AI]")[-1].strip()
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=answer))]
        )

    def _convert_messages_to_prompt(self, messages) -> str:
       prompt = ""
       for message in messages:
           prompt += message.content + "\n"
       return prompt


def load_qlora_llm(
    base_model_path="beomi/KoAlpaca-Polyglot-5.8B",
    adapter_path="/qlora_model_koalpaca"
):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    return LocalHuggingFaceChat(model=model, tokenizer=tokenizer)