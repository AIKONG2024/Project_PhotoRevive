import subprocess
import openai
import datetime
import os
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경 변수 읽기
openai_api_key = os.getenv('OPENAI_API_KEY')

# OpenAI API 키 설정
openai.api_key = openai_api_key

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def analyze_prompt_with_openai(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",  # gpt-4 또는 gpt-3.5-turbo 사용
        messages=[
            {"role": "system", "content": "You are an assistant that analyzes prompts and lists tasks."},
            {"role": "user", "content": f"""
             이 프롬프트를 분석해서 날씨를 변경하는 프롬프트, 객체를 지우는 프롬프트를 영어로 각각 뽑아내줘. 
             index는 만들지 말고  
             'Change: 변경하려면 탐지해야하는 객체이름, 변경할 객체', 'Remove: 변경하려면 탐지해야하는 객체이름, 변경할 객체' 이 형태로 뽑아줘.
             다만 변경할 객체가 날씨(Weather)라면 탐지해야하는 객체이름을 sky로 해줘 : {prompt}\n\nTasks:"""}
        ],
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
    )
    
    tasks = response['choices'][0]['message']['content'].strip().split("\n")
    change_tasks = []
    remove_tasks = []
    for task in tasks:
        for item in task.split(','):
            print(item)
            if 'change' in item.lower():
                print('====', item.split(':')[1])
                change_tasks.append(item.split(':')[1].strip())
            elif 'remove' in item.lower():
                print('=====', item.split(':')[1])
                remove_tasks.append(item.split(':')[1].strip())
    return change_tasks, remove_tasks

# GroundingSAM 및 inpainting을 위한 명령 실행 함수
def run_command(command):
    result = subprocess.run(command, shell=True, check=True, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Error executing command: {command}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, command)
    return result.stdout, result.stderr

def generate_command(task_type, task_detail, image_path, output_dir):
    base_command = "CUDA_VISIBLE_DEVICES=0 python"
    script = "grounded_sam_inpainting_demo_custom_mask.py"
    
    # 스크립트 결정
    if task_type == "remove":
        script = "grounded_sam_remove.py"
    
    # det_prompt와 inpaint_prompt 설정
    det_prompt = task_detail
    inpaint_prompt = task_detail if task_type == "change" else ""

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
    change_tasks, remove_tasks = analyze_prompt_with_openai(prompt)
    
    print(f"Change tasks identified: {change_tasks}")
    print(f"Remove tasks identified: {remove_tasks}")
    
    # Change 작업 수행
    for task_detail in change_tasks:
        command = generate_command("change", task_detail, image_path, output_dir)
        try:
            stdout, stderr = run_command(command)
            print(f"Command output: {stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
            continue

        # 결과 이미지를 다음 작업에 사용할 이미지로 설정
        image_path = os.path.join(output_dir, 'inpainted_image.jpg')
    
    # Remove 작업 수행
    for task_detail in remove_tasks:
        command = generate_command("remove", task_detail, image_path, output_dir)
        try:
            stdout, stderr = run_command(command)
            print(f"Command output: {stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
            continue

        # 결과 이미지를 다음 작업에 사용할 이미지로 설정
        image_path = os.path.join(output_dir, 'output_image.jpg')

if __name__ == "__main__":
    prompt = "날씨를 번개 치게 변경하고, 사람들을 지워줘"
    image_path = './assets/test3.jpg'
    main_workflow(prompt, image_path)