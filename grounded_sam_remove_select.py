import argparse
import os
import subprocess
import numpy as np
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import datetime
from segment_anything import build_sam, SamPredictor

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Grounded-Segment-Anything', 'GroundingDINO')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Grounded-Segment-Anything')))

import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# 타임스탬프
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def load_image(image_path):
    # 이미지 로드
    image_pil = Image.open(image_path).convert("RGB")  # 이미지 로드

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def area(box):
    x_min, y_min, x_max, y_max = box
    return (x_max - x_min) * (y_max - y_min)

def box_inclusion(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inclusion_x_min = max(x1_min, x2_min)
    inclusion_y_min = max(y1_min, y2_min)
    inclusion_x_max = min(x1_max, x2_max)
    inclusion_y_max = min(y1_max, y2_max)

    if inclusion_x_min >= inclusion_x_max or inclusion_y_min >= inclusion_y_max:
        return 0.0  # 포함되는 부분이 없음

    inclusion_area = (inclusion_x_max - inclusion_x_min) * (inclusion_y_max - inclusion_y_min)
    box2_area = area(box2)

    return inclusion_area / box2_area

def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    
    # 출력 필터링
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]  # num_filt, 256
    boxes_filt = boxes[filt_mask]  # num_filt, 4

    # 문구 추출
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")

    return boxes_filt, pred_phrases

def convert_boxes_format(boxes, image_width, image_height):
    """
    Grounding DINO의 바운딩박스를 좌상단(x_min, y_min)과 우하단(x_max, y_max) 좌표로 변환
    """
    converted_boxes = []
    for box in boxes:
        x_center, y_center, width, height = box
        x_min = (x_center - width / 2) * image_width
        y_min = (y_center - height / 2) * image_height
        x_max = (x_center + width / 2) * image_width
        y_max = (y_center + height / 2) * image_height
        converted_boxes.append([x_min, y_min, x_max, y_max])
    return converted_boxes

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)

def draw_boxes_on_image(image, boxes, color=(0, 255, 0), thickness=2):
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
    return image

def select_exclude_range(image):
    exclude_range = []
    clone = image.copy()

    def select_area(event, x, y, flags, param):
        nonlocal exclude_range, clone
        if event == cv2.EVENT_LBUTTONDOWN:
            exclude_range = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            exclude_range.append((x, y))
            cv2.rectangle(clone, exclude_range[0], exclude_range[1], (0, 255, 0), 2)
            cv2.imshow("Select Exclude Range", clone)
        elif event == cv2.EVENT_MOUSEMOVE and len(exclude_range) == 1:
            # 실시간으로 박스를 그리기 위해 이동 중일 때
            temp_clone = clone.copy()
            cv2.rectangle(temp_clone, exclude_range[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("Select Exclude Range", temp_clone)

    window_name = "Select Exclude Range"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)  # 필요에 따라 창 크기 조정
    cv2.imshow(window_name, image)
    cv2.setMouseCallback(window_name, select_area)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(exclude_range) == 2:
        x_min = min(exclude_range[0][0], exclude_range[1][0])
        y_min = min(exclude_range[0][1], exclude_range[1][1])
        x_max = max(exclude_range[0][0], exclude_range[1][0])
        y_max = max(exclude_range[0][1], exclude_range[1][1])
        return x_min, y_min, x_max, y_max
    else:
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--grounded_checkpoint", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--det_prompt", type=str, required=True, help="text prompt")
    parser.add_argument("--inpaint_prompt", type=str, required=True, help="inpaint prompt")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")
    parser.add_argument("--cache_dir", type=str, default=None, help="save your huggingface large model cache")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--inpaint_mode", type=str, default="first", help="inpaint mode")
    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # 모델 설정 파일 경로 변경
    grounded_checkpoint = args.grounded_checkpoint  # 모델 경로 변경
    sam_checkpoint = args.sam_checkpoint
    image_path = args.input_image
    det_prompt = args.det_prompt
    inpaint_prompt = args.inpaint_prompt
    output_dir = args.output_dir + f"/removed/"
    cache_dir = args.cache_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    inpaint_mode = args.inpaint_mode
    device = args.device

    # 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    # 이미지 로드
    image_pil, image = load_image(image_path)
    image_cv = cv2.imread(image_path)

    # 제외 범위 선택
    exclude_range = select_exclude_range(image_cv)

    # 모델 로드
    model = load_model(config_file, grounded_checkpoint, device=device)

    # 원본 이미지 시각화
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # Grounding DINO 모델 실행
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, det_prompt, box_threshold, text_threshold, device=device
    )

    # 선택된 제외 범위와 포함 비율을 기반으로 박스 제외
    if exclude_range is not None:
        x_min, y_min, x_max, y_max = exclude_range
        exclude_mask = []

        # PIL 이미지 객체의 크기를 가져옴
        width, height = image_pil.size  # 이미지의 실제 크기
        # 박스 좌표를 이미지의 크기에 맞게 변환
        converted_boxes_filt = convert_boxes_format(boxes_filt, width, height)
        for box in converted_boxes_filt:
            print(box)
            inclusion_ratio = box_inclusion([x_min, y_min, x_max, y_max], box)
            print("Inclusion Ratio:", inclusion_ratio)
            exclude_mask.append(inclusion_ratio < 0.6)  # 포함 비율이 0.6 이상인 박스를 제외
        exclude_mask = torch.tensor(exclude_mask, dtype=torch.bool)
        boxes_filt = boxes_filt[exclude_mask]
        pred_phrases = [phrase for i, phrase in enumerate(pred_phrases) if exclude_mask[i]]

    # SAM 초기화
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False,
    )
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "grounded_sam_output.jpg"), bbox_inches="tight")

    # 마스크 병합
    masks = torch.sum(masks, dim=0).unsqueeze(0)
    masks = torch.where(masks > 0, True, False)
    mask = masks[0][0].cpu().numpy()  # 간단히 첫 번째 마스크 선택, 이는 추후 릴리즈에서 정제될 예정
    mask_pil = Image.fromarray(mask)
    image_pil = Image.fromarray(image)

    # 마스크를 흑백 이미지로 저장
    bw_mask = (masks[0][0].cpu().numpy() * 255).astype(np.uint8)
    bw_mask_pil = Image.fromarray(bw_mask, mode="L")
    bw_mask_pil.save(os.path.join(output_dir, "remove_mask_bw.jpg"))

    kernel_size_row = 18
    kernel_size_col = 18
    kernel = np.ones((kernel_size_row, kernel_size_col), np.uint8)

    dilation_image = cv2.dilate(bw_mask, kernel, iterations=1)  # 팽창 이미지 생성
    dilated_mask_pil = Image.fromarray(dilation_image, mode="L")
    dilated_mask_pil.save(os.path.join(output_dir, "dilated_mask_bw.jpg"))

    # 인페인팅 파이프라인
    script2 = f"""
    iopaint run --model=zits --device=cpu --image={output_dir + "raw_image.jpg"} --mask={output_dir + "dilated_mask_bw.jpg"} --output={output_dir}
    """

    def run(command):
        result = subprocess.run(command, shell=True, check=True, encoding="utf-8")
        return result

    run(script2)
