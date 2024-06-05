import os
from dotenv import load_dotenv
import openai
import numpy as np
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from sklearn.metrics.pairwise import cosine_similarity

# .env 파일 로드
load_dotenv()

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

# GPT-3.5 Turbo 모델 로드
llm = openai.ChatCompletion

# Few-shot 학습을 위한 예제 데이터
examples = [
    {"input": "오늘 날씨가 좋다.", "label": "None", "task": "None"},
    {"input": "오늘 날씨가 좋고 사람들이 많네.", "label": "None", "task": "None"},
    {"input": "날씨가 왜이래? 사람들이 왜 이렇게 많아?.", "label": "None", "task": "None"},
    {"input": "나는 사과를 먹었다.", "label": "None", "task": "None"},
    {"input": "비가 오는 날씨로 바꿔야 한다.", "label": "날씨 변경", "det_prompt": "sky", "inpainting_prompt": "rainy sky", "task": "remove"},
    {"input": "이 사진에서 자동차를 제거해야 한다.", "label": "객체 제거", "det_prompt": "car", "inpainting_prompt": "remove car", "task": "remove"},
    {"input": "천둥치는 날씨로 바꿔야 한다.", "label": "날씨 변경", "det_prompt": "sky", "inpainting_prompt": "thunder sky", "task" : "change"},
    {"input": "맑은 날씨로 바꿔야 한다.", "label": "날씨 변경", "det_prompt": "sky", "inpainting_prompt": "clear sky", "task" : "change"},
    {"input": "사람을 제거해줘.", "label": "객체 제거", "det_prompt": "person", "inpainting_prompt": "remove person", "task": "remove object", "task" : "remove"},
    {"input": "여기 나온 사람들 제거해줘.", "label": "객체 제거", "det_prompt": "persons", "inpainting_prompt": "remove persons", "task": "remove object", "task" : "remove"},
    {"input": "물컵을 제거해줘.", "label": "객체 제거", "det_prompt": "cup", "inpainting_prompt": "remove cup", "task": "remove object", "task" :"remove"},
    {"input": "사람들을 제거 해주고 맑은 하늘로 변경해줘.", "label": "객체 제거, 날씨 변경", "det_prompt": "person, sky", "inpainting_prompt": "remove person, clear sky", "task": "remove object, change"},
    {"input": "번개치는 하늘로 변경해주고 사람들을 제거 해줘.", "label": "날씨 변경, 객체 제거", "det_prompt": "sky, person", "inpainting_prompt": "a lightening sky,remove person", "task": "change, remove object"}
]

# Few-shot 프롬프트 템플릿 정의
prompt_template = """
{input}

이 문장에서 '날씨 변경'이나 '객체 제거' 작업이 필요한지 답해주세요. 필요하다면 해당 작업을, 아니라면 'None'이라고 답해주세요. 또한, 날씨 변경 작업이 필요한 경우 'det prompt: [det_prompt]. inpainting prompt: [inpainting_prompt].', 객체 제거 작업이 필요한 경우 'det prompt: [det_prompt]. inpainting prompt: remove [det_prompt].' 형식으로 작성해주세요.
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["input", "det_prompt", "inpainting_prompt", "task"])
few_shot_prompt = FewShotPromptTemplate(
    example_prompt=prompt,
    examples=examples,
    prefix="다음은 Few-shot 학습을 위한 예제입니다.",
    suffix="이제 새로운 문장에 대해 예측해 보겠습니다.",
    example_separator="\n\n",
    input_variables=["input"]
)

# OpenAI Embedding 함수
def get_embeddings(texts):
    response = openai.Embedding.create(input=texts, model="text-embedding-ada-002")
    return [embedding['embedding'] for embedding in response['data']]

def process_user_input(user_input):
    # Few-shot 학습 및 예측
    few_shot_examples = "\n\n".join([f"{ex['input']}\ndet prompt: {ex['det_prompt']}. inpainting prompt: {ex['inpainting_prompt']}. task: {ex['task']}" for ex in examples if ex['label'] != "None"])
    prompt_with_examples = f"{few_shot_examples}\n\n{user_input}\n"
    
    response = llm.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful prompt assistant."},
            {"role": "user", "content": prompt_with_examples}
        ],
        temperature=0.0,
        max_tokens=100,
        n=1,
        stop=None
    )
    prediction = response.choices[0].message.content.strip()

    # 텍스트 임베딩을 통한 유사도 계산
    user_embedding = get_embeddings([user_input])[0]
    task_embeddings = get_embeddings(['날씨 변경', '객체 제거'])
    similarities = cosine_similarity([user_embedding], task_embeddings)[0]

    # 유사도가 높은 작업에 따른 프롬프트 생성
    if max(similarities) > 0.5:
        task_index = np.argmax(similarities)
        task = ['날씨 변경', '객체 제거'][task_index]
        
        det_prompt, inpainting_prompt = extract_prompts_from_prediction(prediction, task)

        if task == '날씨 변경':
            detection_prompt = f"다음 이미지에서 {det_prompt}를 감지해야 합니다: {user_input}"
            inpainting_prompt = f"이전 이미지에서 감지된 {det_prompt}를 {inpainting_prompt}로 변경해야 합니다."
        else:
            detection_prompt = f"다음 이미지에서 {det_prompt}를 감지해야 합니다: {user_input}"
            inpainting_prompt = f"이전 이미지에서 감지된 {det_prompt}를 제거해야 합니다."
        
        return detection_prompt, inpainting_prompt, prediction
    else:
        return None, None, prediction

def extract_prompts_from_prediction(prediction, task):
    lines = prediction.split(". ")
    det_prompt = None
    inpainting_prompt = None
    for line in lines:
        if line.startswith("det prompt:"):
            det_prompt = line.replace("det prompt:", "").strip()
        if line.startswith("inpainting prompt:"):
            inpainting_prompt = line.replace("inpainting prompt:", "").strip()
        if line.startswith("inpainting prompt:"):
            inpainting_prompt = line.replace("inpainting prompt:", "").strip()
    return det_prompt, inpainting_prompt

# 예시로 사용자 입력을 처리하는 코드
user_input = "불이나서 잿빞이 흩날리는 날이나 화산이 폭발해서 재가 날리며 불난 것 처럼 변경 해주고 사람들이 모두 사라졌으면 좋겠어."
detection_prompt, inpainting_prompt, prediction = process_user_input(user_input)

if detection_prompt and inpainting_prompt:
    print(f"Detection 프롬프트: {detection_prompt}")
    print(f"Inpainting 프롬프트: {inpainting_prompt}")
else:
    print("유사한 작업이 없습니다.")

print(f"GPT 모델 출력: {prediction}")
