from flask import Flask, request, render_template, url_for, redirect
import os
import subprocess
from langchain_task import LangChain
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['OUTPUT_FOLDER'] = 'static/outputs/'

def analyze_prompt(prompt):
    return LangChain().process_user_input(prompt)

def run_command(command):
    subprocess.run(command, shell=True, check=True, encoding="utf-8")

def generate_command(task, image_path, output_dir, bbox):
    if task['label'] == "날씨 변경":
        command = generate_weather_change_command(task, image_path, output_dir)
    elif task['label'] == "객체 제거":
        command = generate_object_removal_command(task, image_path, output_dir, bbox)
    return command

def generate_weather_change_command(task, image_path, output_dir):
    script = "grounded_sam_inpainting_2_demo_custom_mask.py"
    det_prompt = task['det_prompt']
    inpaint_prompt = task['inpainting_prompt']
    
    command = f"""
    CUDA_VISIBLE_DEVICES=0 python {script} \
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

def generate_object_removal_command(task, image_path, output_dir, bbox):
    script = "grounded_sam_remove_select.py"
    det_prompt = task['det_prompt']
    inpaint_prompt = task['inpainting_prompt']
    
    command = f"""
    CUDA_VISIBLE_DEVICES=0 python {script} \
    --config Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --grounded_checkpoint Grounded-Segment-Anything/groundingdino_swint_ogc.pth \
    --sam_checkpoint Grounded-Segment-Anything/sam_vit_h_4b8939.pth \
    --input_image {image_path} \
    --output_dir {output_dir} \
    --box_threshold 0.2 \
    --text_threshold 0.5 \
    --det_prompt "{det_prompt}" \
    --inpaint_prompt "{inpaint_prompt}" \
    --device "cuda" \
    --bbox "{bbox}"
    """
    return command

def main_workflow(prompt, image_path, bbox):
    output_dir = app.config['OUTPUT_FOLDER']

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tasks = analyze_prompt(prompt)
    tasks = json.loads(tasks)
    print(tasks)

    image_paths = [image_path]  # 각 작업의 결과 이미지를 저장할 리스트

    for task in tasks['tasks']:
        command = generate_command(task, image_paths[-1], output_dir, bbox)
        run_command(command)
        if task['label'] == "날씨 변경":
           image_path = "inpainted_image.jpg"
        elif task['label'] == "객체 제거":
           image_path = "raw_image.png"

    return image_path, tasks['tasks']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    prompt = request.form['prompt']
    bbox = request.form.get('bbox', "0.0,0.0,0.0,0.0")  # 기본값 설정
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            print(f"BBox: {bbox}")  # 바운딩 박스 출력
            return redirect(url_for('process', filename=filename, prompt=prompt, bbox=bbox))
        except Exception as e:
            return f"An error occurred: {e}"

@app.route('/process')
def process():
    filename = request.args.get('filename')
    prompt = request.args.get('prompt')
    bbox = request.args.get('bbox')
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        output_image, tasks = main_workflow(prompt, file_path, bbox)
        print(output_image)
        return render_template('result.html', filename=filename, prompt=prompt, output_image=output_image, tasks=tasks)
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['OUTPUT_FOLDER']):
        os.makedirs(app.config['OUTPUT_FOLDER'])
    app.run(host='0.0.0.0', port=9000)