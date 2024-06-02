import subprocess
import datetime
import os
import spacy

# spaCy 모델 로드
nlp = spacy.load("ko_core_news_sm")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# LLM을 사용하여 프롬프트 분석 및 작업 분리
def analyze_prompt(prompt):
    weather_tasks = set()  # 중복 작업 방지를 위해 set 사용
    person_tasks = set()
    doc = nlp(prompt)
    
    print("Analyzing prompt:")
    for sent in doc.sents:
        print(f"Sentence: {sent.text}")
        for token in sent:
            print(f"Token: {token.text}, Lemma: {token.lemma_}, POS: {token.pos_}, Head: {token.head.text}, Dependency: {token.dep_}")
            if token.pos_ == "VERB":  # 동사인 경우 작업 추출
                if "하늘" in sent.text or "날씨" in sent.text:
                    if "맑음" in sent.text or "맑은 하늘" in sent.text or "맑고" in sent.text or "화창" in sent.text:
                        weather_tasks.add("clear_sky")
                    elif "구름" in  sent.text:
                        weather_tasks.add("cloudy_sky")
                    elif "흐림"  in sent.text or "흐린" in sent.text:
                        weather_tasks.add("overcast_sky")
                    elif "번개"  in sent.text or "천둥" in sent.text:
                        weather_tasks.add("stormy_sky")
                    elif "비"  in sent.text:
                        weather_tasks.add("rainy_sky")
                    elif "눈"  in sent.text:
                        weather_tasks.add("snowy_sky")
                    elif "무지개"  in sent.text:
                        weather_tasks.add("rainbow_sky")
                if "사람"  in sent.text or "사람들"  in sent.text:
                    if "중앙 사람 빼고"  in sent.text:
                        person_tasks.add("remove_other_people")
                    else:
                        person_tasks.add("remove_people")
    return list(weather_tasks), list(person_tasks)

# GroundingSAM 및 inpainting을 위한 명령 실행 함수
def run_command(command):
    subprocess.run(command, shell=True, check=True, encoding="utf-8")

def generate_command(task, image_path, output_dir):
    base_command = "CUDA_VISIBLE_DEVICES=0 python"
    script = "grounded_sam_inpainting_demo_custom_mask.py"
    det_prompt = ""
    inpaint_prompt = ""
    
    if task == "clear_sky":
        det_prompt = "sky"
        inpaint_prompt = "A bright day clear blue sky."
    elif task == "cloudy_sky":
        det_prompt = "sky"
        inpaint_prompt = "A cloudy sky."
    elif task == "overcast_sky":
        det_prompt = "sky"
        inpaint_prompt = "An overcast sky."
    elif task == "stormy_sky":
        det_prompt = "sky"
        inpaint_prompt = "A stormy sky with lightning."
    elif task == "rainy_sky":
        det_prompt = "sky"
        inpaint_prompt = "A rainy sky."
    elif task == "snowy_sky":
        det_prompt = "sky"
        inpaint_prompt = "A snowy sky."
    elif task == "rainbow_sky":
        det_prompt = "sky"
        inpaint_prompt = "A sky with a rainbow."
    elif task == "remove_people":
        script = "grounded_sam_remove.py"
        det_prompt = "person"
        inpaint_prompt = ""
    elif task == "remove_other_people":
        script = "grounded_sam_remove.py"
        det_prompt = "other_people"
        inpaint_prompt = ""

    command = f"""
    {base_command} {script} \
    --config Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --grounded_checkpoint Grounded-Segment-Anything/groundingdino_swint_ogc.pth \
    --sam_checkpoint Grounded-Segment-Anything/sam_vit_h_4b8939.pth \
    --input_image {image_path} \
    --output_dir {output_dir} \
    --box_threshold 0.7 \
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
    weather_tasks, person_tasks = analyze_prompt(prompt)
    
    print(f"Weather tasks identified: {weather_tasks}")
    print(f"Person tasks identified: {person_tasks}")
    
    # 날씨 관련 작업을 먼저 수행
    for task in weather_tasks:
        command = generate_command(task, image_path, output_dir)
        run_command(command)
        image_path = os.path.join(output_dir, "inpainted_image.jpg")
    
    # 사람 제거 작업을 나중에 수행
    for task in person_tasks:
        command = generate_command(task, image_path, output_dir)
        run_command(command)
        image_path = os.path.join(output_dir, "output_image.jpg")

if __name__ == "__main__":
    prompt = "맑은 하늘로 변경해주고, 중앙 사람 빼고 뒤에 나머지 사람들을 지워줘"
    image_path = './assets/test100.jpg'
    main_workflow(prompt, image_path)