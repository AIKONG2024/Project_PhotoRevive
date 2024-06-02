import subprocess
import datetime
import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import openai
import os

# .env 파일 로드
load_dotenv()

# 환경 변수 읽기
openai_api_key = os.getenv('OPENAI_API_KEY')

# OpenAI API 키 설정
openai.api_key = openai_api_key
llm = OpenAI(api_key=openai_api_key, model="text-davinci-003")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def analyze_prompt_with_langchain(prompt):
    prompt_template = PromptTemplate.from_template("Analyze the following prompt and list tasks:\n\nPrompt: {prompt}\n\nTasks:")
    formatted_prompt = prompt_template.format(prompt=prompt)
    
    response = llm(formatted_prompt)
    tasks = response.strip().split("\n")
    tasks = [task.strip() for task in tasks if task.strip()]
    return tasks

# GroundingSAM 및 inpainting을 위한 명령 실행 함수
def run_command(command):
    subprocess.run(command, shell=True, check=True, encoding="utf-8")

def generate_command(task, image_path, output_dir):
    base_command = "CUDA_VISIBLE_DEVICES=0 python"
    script = ""
    det_prompt = ""
    inpaint_prompt = ""
    
    if task == "clear_sky":
        script = "grounded_sam_inpainting_demo_custom_mask.py"
        det_prompt = "sky"
        inpaint_prompt = "A bright day clear blue sky."
    elif task == "cloudy_sky":
        script = "grounded_sam_inpainting_demo_custom_mask.py"
        det_prompt = "sky"
        inpaint_prompt = "A cloudy sky."
    elif task == "overcast_sky":
        script = "grounded_sam_inpainting_demo_custom_mask.py"
        det_prompt = "sky"
        inpaint_prompt = "An overcast sky."
    elif task == "stormy_sky":
        script = "grounded_sam_inpainting_demo_custom_mask.py"
        det_prompt = "sky"
        inpaint_prompt = "A stormy sky with lightning."
    elif task == "rainy_sky":
        script = "grounded_sam_inpainting_demo_custom_mask.py"
        det_prompt = "sky"
        inpaint_prompt = "A rainy sky."
    elif task == "snowy_sky":
        script = "grounded_sam_inpainting_demo_custom_mask.py"
        det_prompt = "sky"
        inpaint_prompt = "A snowy sky."
    elif task == "rainbow_sky":
        script = "grounded_sam_inpainting_demo_custom_mask.py"
        det_prompt = "sky"
        inpaint_prompt = "A sky with a rainbow."
    elif task == "remove_people":
        script = "grounded_sam_remove.py"
        det_prompt = "person"
        inpaint_prompt = ""

    command = f"""
    {base_command} {script} \
    --config Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --grounded_checkpoint Grounded-Segment-Anything/groundingdino_swint_ogc.pth \
    --sam_checkpoint Grounded-Segment-Anything/sam_vit_h_4b8939.pth \
    --input_image {image_path} \
    --output_dir {output_dir} \
    --box_threshold 0.2 \
    --text_threshold 0.5 \
    --det_prompt "{det_prompt}" \
    --inpaint_prompt "{inpaint_prompt}" \
    --device "cuda"
    """
    return command

def main_workflow(prompt, image_path):
    
    output_dir = f"./outputs/{timestamp}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 프롬프트 분석 및 작업 분리
    tasks = analyze_prompt_with_langchain(prompt)
    
    print(f"Tasks identified: {tasks}")
    
    # 각 작업 수행
    for task in tasks:
        command = generate_command(task, image_path, output_dir)
        run_command(command)
        # 결과 이미지를 다음 작업에 사용할 이미지로 설정
        if task in ["clear_sky", "cloudy_sky", "overcast_sky", "stormy_sky", "rainy_sky", "snowy_sky", "rainbow_sky"]:
            image_path = os.path.join(output_dir, "inpainted_image.jpg")
        elif task == "remove_people":
            image_path = os.path.join(output_dir, "output_image.jpg")

if __name__ == "__main__":
    prompt = "날씨를 번개 치게 변경하고, 사람들을 지워줘"
    image_path = './assets/test.jpg'
    main_workflow(prompt, image_path)