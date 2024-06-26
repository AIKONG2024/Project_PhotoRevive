import os
import json
import time
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

# .env 파일 로드
load_dotenv()

class LangChain:
    def __init__(self):
        self.examples = [
            {"input": "오늘 날씨가 좋다.", "tasks": []},
            {"input": "오늘 날씨가 좋고 사람들이 많네.", "tasks": []},
            {"input": "날씨가 왜이래? 사람들이 왜 이렇게 많아?.", "tasks": []},
            {"input": "나는 사과를 먹었다.", "tasks": []},
            {"input": "비가 오는 날씨로 바꿔야 한다.", "tasks": [{"label": "날씨 변경", "det_prompt": "sky", "inpainting_prompt": "a rainy day sky"}]},
            {"input": "이 사진에서 자동차를 제거해야 한다.", "tasks": [{"label": "객체 제거", "det_prompt": "car", "inpainting_prompt": "remove car"}]},
            {"input": "천둥치는 날씨로 바꿔야 한다.", "tasks": [{"label": "날씨 변경", "det_prompt": "sky", "inpainting_prompt": "a thundering day sky"}]},
            {"input": "맑은 날씨로 바꿔야 한다.", "tasks": [{"label": "날씨 변경", "det_prompt": "sky", "inpainting_prompt": "a clear day blue sky"}]},
            {"input": "맑은 하늘로 변경해야 한다.", "tasks": [{"label": "날씨 변경", "det_prompt": "sky", "inpainting_prompt": "a clear day blue sky"}]},
            {"input": "사람을 제거해줘.", "tasks": [{"label": "객체 제거", "det_prompt": "person", "inpainting_prompt": "remove person"}]},
            {"input": "여기 나온 사람들 제거해줘.", "tasks": [{"label": "객체 제거", "det_prompt": "persons", "inpainting_prompt": "remove persons"}]},
            {"input": "물컵을 제거해줘.", "tasks": [{"label": "객체 제거", "det_prompt": "cup", "inpainting_prompt": "remove cup"}]},
            {"input": "사람들을 제거 해주고 맑은 하늘로 변경해줘.", "tasks": [{"label": "객체 제거", "det_prompt": "person", "inpainting_prompt": "remove person"}, {"label": "날씨 변경", "det_prompt": "sky", "inpainting_prompt": "clear sky"}]},
            {"input": "번개치는 하늘로 변경해주고 사람들을 제거 해줘.", "tasks": [{"label": "날씨 변경", "det_prompt": "sky", "inpainting_prompt": "a lightning day sky"}, {"label": "객체 제거", "det_prompt": "person", "inpainting_prompt": "remove person"}]},
            {"input": "비오는 하늘로 변경해주고 사람들을 제거 해줘.", "tasks": [{"label": "날씨 변경", "det_prompt": "sky", "inpainting_prompt": "a rainy day sky"}, {"label": "객체 제거", "det_prompt": "person", "inpainting_prompt": "remove person"}]},
            {"input": "지나가는 사람을 없애고, 맑은 하늘로 변경.", "tasks": [{"label": "객체 제거", "det_prompt": "person", "inpainting_prompt": "remove person"}, {"label": "날씨 변경", "det_prompt": "sky", "inpainting_prompt": "a clear day blue sky"}]},
            {"input": "매우 맑고 화창한 하늘로 변경해주고, 사람들을 제거 해줘.", "tasks": [{"label": "날씨 변경", "det_prompt": "sky", "inpainting_prompt": "a clear day blue sky"}, {"label": "객체 제거", "det_prompt": "person", "inpainting_prompt": "remove person"}]},
            {"input": "우산과 사람을 제거 해줘.", "tasks": [{"label": "객체 제거", "det_prompt": "umbrella, person", "inpainting_prompt": "remove umbrella,person"}]},
            {"input": "사람을 제거해주고, 모자 제거 해줘.", "tasks": [{"label": "객체 제거", "det_prompt": "person, hat", "inpainting_prompt": "remove person,hat"}]},
            {"input": "맑고 푸른 하늘로 바꿔줘.", "tasks": [{"label": "날씨 변경", "det_prompt": "sky", "inpainting_prompt": "a clear azure sky with soft, wispy clouds"}]},
        ]

        self.prompt_template = """
        다음은 사용자 입력과 해당 작업을 JSON 형식으로 나타낸 예제입니다:
        예제:
        {examples}
        이제 사용자의 입력을 JSON 형식으로 처리하세요:
        사용자 입력: {input}
        날씨 변경 inpainting_prompt를 좀 더 하늘을 잘 생성하도록 정확하고 구체적으로 생성해줘.
        사람 제거에 대한 프롬프트도 잘 감지해줘.
        JSON 형식으로 출력:
        """

        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
        self.chain = LLMChain(prompt=PromptTemplate(input_variables=["examples", "input"], template=self.prompt_template), llm=self.llm)

    def process_user_input(self, user_input):
        start_time = time.time()
        examples = "\n".join([
            f'{{"input": "{ex["input"]}", "tasks": {json.dumps(ex["tasks"], ensure_ascii=False)}}}' 
            for ex in self.examples
        ])
        prompt_with_examples = self.prompt_template.format(examples=examples, input=user_input)
        response = self.chain.run({"examples": examples, "input": user_input})
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.2f} seconds")
        
        try:
            parsed_result = json.loads(response)
        except json.JSONDecodeError:
            print("JSON 파싱에 실패했습니다.")
            parsed_result = {"error": "Invalid JSON format"}

        return json.dumps(parsed_result, ensure_ascii=False)

if __name__ == '__main__':
    lang = LangChain()
    result = lang.process_user_input("매우 맑고 화창한 하늘로 변경해주고, 사람들을 제거 해줘.")
    print(result)
