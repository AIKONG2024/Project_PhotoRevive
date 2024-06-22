import os
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class LangChainExample:
    def __init__(self):
        self.examples = [
            {"input": "비가 오는 날씨로 바꿔야 한다.", "tasks": [{"label": "날씨 변경", "det_prompt": "sky", "inpainting_prompt": "rainy sky"}]},
            {"input": "사람들을 제거해줘.", "tasks": [{"label": "객체 제거", "det_prompt": "person", "inpainting_prompt": "remove person"}]},
            {"input": "맑은 하늘로 변경해주고 사람들을 제거 해줘.", "tasks": [{"label": "날씨 변경", "det_prompt": "sky", "inpainting_prompt": "clear sky"}, {"label": "객체 제거", "det_prompt": "person", "inpainting_prompt": "remove person"}]}
        ]

        self.prompt_template = """
        시스템: 당신은 사용자의 입력을 분석하여 이미지 편집 작업을 JSON 형식으로 출력하는 AI 어시스턴트입니다. 날씨 변경과 객체 제거 작업을 처리할 수 있습니다.

        예제:
        {examples}

        출력:
        사용자: {input}
        어시스턴트:
        """

        # 예제를 템플릿에 맞게 포맷팅
        self.examples_formatted = "\n".join(
            f"사용자: {ex['input']}\n어시스턴트: {json.dumps({'tasks': ex['tasks']}, ensure_ascii=False, indent=2)}"
            for ex in self.examples
        )

        # 모델과 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained("beomi/Llama-3-Open-Ko-8B")
        self.model = AutoModelForCausalLM.from_pretrained("beomi/Llama-3-Open-Ko-8B")

        # HuggingFacePipeline 설정
        hf_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=500
        )
        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)

        # LangChain LLMChain 설정
        self.llm_chain = LLMChain(
            prompt=PromptTemplate(
                input_variables=["input", "examples"],
                template=self.prompt_template
            ),
            llm=self.llm
        )

    def process_user_input(self, user_input):
        start_time = time.time()
        response = self.llm_chain.run({"input": user_input, "examples": self.examples_formatted})
        end_time = time.time()
        print(response)
        
        try:
            # "출력:" 뒤의 내용을 추출
            output_part = response.split("출력:")[-1].strip()
            # "사용자:" 부분을 추출하여 input으로 설정
            user_input = output_part.split("어시스턴트:")[0].replace("사용자:", "").strip()
            # "어시스턴트:" 부분을 추출하여 tasks로 설정
            assistant_response = output_part.split("어시스턴트:")[-1].strip()

            # tasks JSON 파싱
            parsed_tasks = json.loads(assistant_response)

            # 결과 JSON 생성
            result = {
                "input": user_input,
                "tasks": parsed_tasks.get("tasks", [])
            }
            
            execution_time = end_time - start_time
            print(f"Execution time: {execution_time:.2f} seconds")
            return json.dumps(result, ensure_ascii=False, indent=2)
        
        except (json.JSONDecodeError, ValueError, IndexError) as e:
            print(f"JSON 파싱에 실패했습니다: {e}")
            execution_time = end_time - start_time
            print(f"Execution time: {execution_time:.2f} seconds")
            return json.dumps({"error": "Invalid JSON format"}, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    lang = LangChainExample()
    result = lang.process_user_input("매우 맑고 화창한 하늘로 변경해주고, 사람들을 제거 해줘.")
    print(result)
