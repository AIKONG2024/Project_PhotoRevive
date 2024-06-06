from flask import Flask, request, render_template, url_for
import os
import subprocess
import datetime
import spacy
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Spacy 언어 모델 로드
nlp = spacy.load("ko_core_news_sm")

# Sentence Transformers 모델 로드
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 미리 정의된 변경 작업 관련 문장
change_phrases = [
    "변경해줘",
    "바꿔줘",
    "바꾸기",
    "바꾸다",
    "변경하고",
    "바꿔"
]

# 미리 정의된 제거 작업 관련 문장
remove_phrases = [
    "제거",
    "지우기",
    "지워",
    "없애기",
    "지우다"
]

# 변경 작업 관련 문장의 임베딩
change_embeddings = model.encode(change_phrases)
# 제거 작업 관련 문장의 임베딩
remove_embeddings = model.encode(remove_phrases)

def analyze_prompt(prompt):
    change_tasks = set()  # 변경 작업
    remove_tasks = set()  # 제거 작업
    doc = nlp(prompt)
    
    for sent in doc.sents:
        # 현재 문장의 임베딩 계산
        sent_embedding = model.encode(sent.text)

        # 변경 작업과의 유사도 계산
        change_cos_sim = util.pytorch_cos_sim(sent_embedding, change_embeddings)
        # 제거 작업과의 유사도 계산
        remove_cos_sim = util.pytorch_cos_sim(sent_embedding, remove_embeddings)
        print(change_cos_sim)
        print(remove_cos_sim)
        if change_cos_sim.max() >= 0.3:
            if "하늘" in sent.text or "날씨" in sent.text:
                if "맑은" in sent.text or "맑음" in sent.text or "푸른" in sent.text or "화창" in sent.text:
                    change_tasks.add("clear_sky")
                elif "구름" in sent.text:
                    change_tasks.add("cloudy_sky")
                elif "흐림" in sent.text or "흐린" in sent.text:
                    change_tasks.add("overcast_sky")
                elif "번개" in sent.text or "천둥" in sent.text:
                    change_tasks.add("stormy_sky")
                elif "비" in sent.text:
                    change_tasks.add("rainy_sky")
                elif "눈" in sent.text:
                    change_tasks.add("snowy_sky")
                elif "무지개" in sent.text:
                    change_tasks.add("rainbow_sky")
        if remove_cos_sim.max() >= 0.3:
            if "사람" in sent.text or "사람들" in sent.text:
                remove_tasks.add("remove_people")
    
    # 변경 작업이 먼저 오도록 순서를 조정
    tasks = list(change_tasks) + list(remove_tasks)
    return tasks  # 모든 작업 반환

# GroundingSAM 및 inpainting을 위한 명령 실행 함수
def run_command(command):
    subprocess.run(command, shell=True, check=True, encoding="utf-8")

def generate_command(task, image_path, output_dir):
    base_command = "CUDA_VISIBLE_DEVICES=0 python"
    script = "grounded_sam_inpainting_2_demo_custom_mask.py"
    det_prompt = ""
    inpaint_prompt = ""
    
    if task == "clear_sky":
        det_prompt = "sky"
        inpaint_prompt = "A clear blue sky."
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
        script = "grounded_sam_remove_select_cv.py"
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
    output_dir = f"./static/outputs/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 프롬프트 분석 및 작업 분리
    tasks = analyze_prompt(prompt)
    # 각 작업 수행
    for task in tasks:
        command = generate_command(task, image_path, output_dir)
        run_command(command)
        # 결과 이미지를 다음 작업에 사용할 이미지로 설정
        if task in ["clear_sky", "cloudy_sky", "overcast_sky", "stormy_sky", "rainy_sky", "snowy_sky", "rainbow_sky"]:
            image_path = os.path.join(output_dir, "inpainted_image.jpg")
        elif task == "remove_people":
            image_path = os.path.join(output_dir, "output_image.jpg")
    
    return image_path  # output_image 경로 반환

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    prompt = request.form['prompt']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            return render_template('loading.html', filename=filename, prompt=prompt)
        except Exception as e:
            return f"An error occurred: {e}"

@app.route('/process/<filename>/<prompt>')
def process(filename, prompt):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        # Main workflow 실행
        output_image = main_workflow(prompt, file_path)
        
        return render_template('result.html', filename=filename, prompt=prompt, output_image=output_image)
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(host='0.0.0.0', port=9900)