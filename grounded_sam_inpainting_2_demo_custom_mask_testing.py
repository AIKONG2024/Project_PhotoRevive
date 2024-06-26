import argparse
import os
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Grounded-Segment-Anything', 'GroundingDINO')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Grounded-Segment-Anything')))

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt

# diffusers
import PIL
import requests
import torch
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline
import datetime

def load_image(image_path):
    # 이미지 로드
    image_pil = Image.open(image_path).convert("RGB")  # 이미지 로드

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
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

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
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
    logits.shape[0]

    # output 필터링
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # 프레이즈 가져오기
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    # pred 구축
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

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
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--det_prompt", type=str, required=True, help="text prompt")
    parser.add_argument("--inpaint_prompt", type=str, required=True, help="inpaint prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )
    parser.add_argument("--cache_dir", type=str, default=None, help="save your huggingface large model cache")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--inpaint_mode", type=str, default="first", help="inpaint mode")
    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # 설정
    config_file = args.config  # 모델 설정 파일 경로 변경
    grounded_checkpoint = args.grounded_checkpoint  # 모델 체크포인트 경로 변경
    sam_checkpoint = args.sam_checkpoint
    image_path = args.input_image
    det_prompt = args.det_prompt
    inpaint_prompt = args.inpaint_prompt
    output_dir = args.output_dir
    cache_dir = args.cache_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    inpaint_mode = args.inpaint_mode
    device = args.device

    # 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    # 이미지 로드
    image_pil, image = load_image(image_path)
    # 모델 로드
    model = load_model(config_file, grounded_checkpoint, device=device)

    # 원본 이미지 시각화
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # grounding dino 모델 실행
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, det_prompt, box_threshold, text_threshold, device=device
    )

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

  # 모든 마스크를 하나의 마스크로 합치기
    combined_mask = torch.any(masks, dim=0).cpu().numpy()

    # 출력 이미지 그리기
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(combined_mask, plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "grounded_sam_output.jpg"), bbox_inches="tight")

    # 마스크를 흑백 이미지로 저장
    if combined_mask.ndim == 3:
        combined_mask = combined_mask[0]  # 첫 번째 채널 사용
    bw_mask = (combined_mask * 255).astype(np.uint8)
    bw_mask_pil = Image.fromarray(bw_mask, mode="L")
    bw_mask_pil.save(os.path.join(output_dir, "mask_bw.jpg"))
    
    kernel_size_row = 3
    kernel_size_col = 3
    kernel = np.ones((kernel_size_row, kernel_size_col), np.uint8)

    dilation_image = cv2.dilate(bw_mask, kernel, iterations=1)  #// make dilation image
    # dilation_image = cv2.erode(bw_mask, kernel, iterations=1)

    
    dilated_mask_pil = Image.fromarray(dilation_image, mode="L")
    dilated_mask_pil.save(os.path.join(output_dir, "dilated_mask_bw.jpg"))

    # 인페인팅 파이프라인 설정
    inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    ).to(device)

    # 원본 이미지와 마스크 이미지 로드
    init_image = Image.open(os.path.join(output_dir, "raw_image.jpg")).convert("RGB")
    mask_image = Image.open(os.path.join(output_dir, "dilated_mask_bw.jpg")).convert("RGB")

    # 원본 이미지 크기 가져오기
    width, height = init_image.size

    # 높이와 너비를 8로 나누어 떨어지도록 조정
    height = (height // 8) * 8 * 2
    width = (width // 8) * 8 * 2
    # 이미지를 새로운 크기로 리사이즈
    init_image = init_image.resize((width, height), Image.LANCZOS)
    mask_image = mask_image.resize((width, height), Image.LANCZOS)

    # 인페인팅 수행
    # generator = torch.Generator(device=device).manual_seed(7)  # 시드 설정
    inpaint_result = inpaint_pipeline(
        prompt=inpaint_prompt, 
        image=init_image, 
        mask_image=mask_image, 
        height=height, 
        width=width,
        negative_prompt = "sky, no ground, no objects, no plants, unrealistic sky, oversaturated",
        num_inference_steps= 100, 
        strength=0.86,
        guidance_scale=9.0, 
        # generator=generator
    )
    inpainted_image = inpaint_result.images[0]
    inpainted_image.save(os.path.join(output_dir, "inpainted_image.jpg"))
