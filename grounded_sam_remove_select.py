import argparse
import os
import subprocess
import numpy as np
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from segment_anything import build_sam, SamPredictor

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Grounded-Segment-Anything', 'GroundingDINO')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Grounded-Segment-Anything')))

import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

def load_image(image_path, output_dir):
    image_pil = Image.open(image_path).convert("RGB")
    
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
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

def get_grounding_output(model, image, caption, box_threshold, text_threshold, device="cpu"):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]
    boxes = outputs["pred_boxes"].cpu()[0]
    
    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")

    return boxes_filt, pred_phrases

def convert_boxes_format(boxes, image_width, image_height):
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

def box_inclusion(box1, box2):
    x1_min, y1_min, x1_max, y1_max = map(float, box1)
    x2_min, y2_min, x2_max, y2_max = map(float, box2)

    inclusion_x_min = max(x1_min, x2_min)
    inclusion_y_min = max(y1_min, y2_min)
    inclusion_x_max = min(x1_max, x2_max)
    inclusion_y_max = min(y1_max, y2_max)

    if inclusion_x_min >= inclusion_x_max or inclusion_y_min >= inclusion_y_max:
        return 0.0  # 포함되는 부분이 없음

    inclusion_area = (inclusion_x_max - inclusion_x_min) * (inclusion_y_max - inclusion_y_min)
    box2_area = area([x2_min, y2_min, x2_max, y2_max])

    return inclusion_area / box2_area

def area(box):
    x_min, y_min, x_max, y_max = map(float, box)
    return (x_max - x_min) * (y_max - y_min)

def main(args):
    image_pil, image = load_image(args.input_image, args.output_dir)
    model = load_model(args.config, args.grounded_checkpoint, device=args.device)

    boxes_filt, pred_phrases = get_grounding_output(
        model, image, args.det_prompt, args.box_threshold, args.text_threshold, device=args.device
    )

    # 바운딩 박스 좌표 가져오기 (좌상단 x, y, 우하단 x, y 형식으로 변환)
    if args.bbox:
        print(args.bbox)
        x_min, y_min, width, height = map(float, args.bbox.split(','))
        x_max = x_min + width
        y_max = y_min + height
        exclude_mask = []
        print([x_min, y_min, x_max, y_max])
        img_width, img_height = image_pil.size
        converted_boxes_filt = convert_boxes_format(boxes_filt, img_width, img_height)
        for box in converted_boxes_filt:
            print(f"Converted box: {box}")
            inclusion_ratio = box_inclusion([x_min, y_min, x_max, y_max], box)
            print(f"Inclusion ratio: {inclusion_ratio}")
            exclude_mask.append(inclusion_ratio < 0.6)  # 포함 비율이 0.6 이상
        exclude_mask = torch.tensor(exclude_mask, dtype=torch.bool)
        boxes_filt = boxes_filt[exclude_mask]
        pred_phrases = [phrase for i, phrase in enumerate(pred_phrases) if exclude_mask[i]]

    if boxes_filt.size(0) == 0:
        print("No boxes left after exclusion.")
        return  # 더 이상의 처리가 필요 없음

    predictor = SamPredictor(build_sam(checkpoint=args.sam_checkpoint).to(args.device))
    image = cv2.imread(args.input_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(args.device)

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(args.device),
        multimask_output=False,
    )

    if masks.shape[0] == 0:
        print("No masks generated.")
        return  # 마스크가 생성되지 않은 경우 예외 처리

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    plt.axis('off')
    plt.savefig(os.path.join(args.output_dir, "grounded_sam_output.jpg"), bbox_inches="tight")

    # 마스크 병합
    masks = torch.sum(masks, dim=0).unsqueeze(0)
    masks = torch.where(masks > 0, True, False)
    mask = masks[0][0].cpu().numpy()
    mask_pil = Image.fromarray(mask)
    image_pil = Image.fromarray(image)

    # 마스크를 흑백 이미지로 저장
    bw_mask = (masks[0][0].cpu().numpy() * 255).astype(np.uint8)
    bw_mask_pil = Image.fromarray(bw_mask, mode="L")
    bw_mask_pil.save(os.path.join(args.output_dir, "remove_mask_bw.jpg"))

    kernel_size_row = 18
    kernel_size_col = 18
    kernel = np.ones((kernel_size_row, kernel_size_col), np.uint8)

    dilation_image = cv2.dilate(bw_mask, kernel, iterations=1)
    dilated_mask_pil = Image.fromarray(dilation_image, mode="L")
    dilated_mask_pil.save(os.path.join(args.output_dir, "dilated_mask_bw.jpg"))

    # 인페인팅 파이프라인
    script2 = f"""
    iopaint run --model=lama --device=cpu --image={os.path.join(args.output_dir, "inpainted_image.jpg")} --mask={os.path.join(args.output_dir, "dilated_mask_bw.jpg")} --output={args.output_dir}
    """
    
    def run(command):
        result = subprocess.run(command, shell=True, check=True, encoding="utf-8")
        return result

    run(script2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--grounded_checkpoint", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--det_prompt", type=str, required=True, help="text prompt")
    parser.add_argument("--inpaint_prompt", type=str, required=True, help="inpaint prompt")
    parser.add_argument("--output_dir", type=str, required=True, help="output directory")
    parser.add_argument("--box_threshold", type=float, default=0.2, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.5, help="text threshold")
    parser.add_argument("--bbox", type=str, help="bounding box coordinates")
    parser.add_argument("--device", type=str, default="cuda", help="device to use for inference")
    args = parser.parse_args()

    main(args)
