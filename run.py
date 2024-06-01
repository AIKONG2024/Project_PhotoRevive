import subprocess
import datetime
import os
import spacy

# spaCy 모델 로드
nlp = spacy.load("ko_core_news_sm")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# LLM을 사용하여 프롬프트 분석 및 작업 분리
def analyze_prompt(prompt):
    tasks = []
    doc = nlp(prompt)
    
    print("Analyzing prompt:")
    for sent in doc.sents:
        print(f"Sentence: {sent.text}")
        for token in sent:
            print(f"Token: {token.text}, Lemma: {token.lemma_}, POS: {token.pos_}, Head: {token.head.text}, Dependency: {token.dep_}")
            if token.pos_ == "VERB":  # 동사인 경우 작업 추출
                if "하늘" in sent.text or "날씨" in sent.text:
                    if "맑음" in sent.text or "맑은 하늘" in sent.text or "맑고" in sent.text or "화창" in sent.text:
                        tasks.append("clear_sky")
                    elif "구름" in sent.text:
                        tasks.append("cloudy_sky")
                    elif "흐림" in sent.text or "흐린" in sent.text:
                        tasks.append("overcast_sky")
                    elif "번개" in sent.text or "천둥" in sent.text:
                        tasks.append("stormy_sky")
                    elif "비" in sent.text:
                        tasks.append("rainy_sky")
                    elif "눈" in sent.text:
                        tasks.append("snowy_sky")
                    elif "무지개" in sent.text:
                        tasks.append("rainbow_sky")
                if "사람" in sent.text or "사람들" in sent.text:
                    tasks.append("remove_people")
    return tasks 

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
    tasks = analyze_prompt(prompt)
    
    print(f"Tasks identified: {tasks}")
    
    # 각 작업 수행
    for task in tasks:
        command = generate_command(task, image_path, output_dir)
        run_command(command)
        # 결과 이미지를 다음 작업에 사용할 이미지로 설정
        if task in ["clear_sky", "cloudy_sky", "overcast_sky", "stormy_sky", "rainy_sky", "snowy_sky", "rainbow_sky"]:
            image_path = os.path.join(output_dir, "inpainted_image.jpg")
        elif task == "remove_people":
            image_path = os.path.join(output_dir, "inpainted_image.jpg")

if __name__ == "__main__":
    prompt = "날씨를 천둥번개 치는 날로 변경해주고, 사람들을 지워줘"
    image_path = './assets/test3.jpg'
    main_workflow(prompt, image_path)