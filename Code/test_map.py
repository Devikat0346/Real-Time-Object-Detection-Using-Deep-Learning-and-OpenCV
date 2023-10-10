import argparse
import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import cv2
from detection import YOLOv8ONNX
from tqdm import tqdm
import onnxruntime as ort


def main(model_path, val_dir, annotation_file, save_viz):
    coco_gt = COCO(annotation_file)
    cat_ids = coco_gt.getCatIds()
    img_ids = coco_gt.getImgIds()
    device = ort.get_device()
    print("using device:", device)
    yolo = YOLOv8ONNX(model_path, device = device)
    results = []

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    viz_output_dir = f"viz_output_{model_name}"
    if save_viz:
        os.makedirs(viz_output_dir, exist_ok=True)

    for img_id in tqdm(img_ids):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(val_dir, img_info["file_name"])
        image = cv2.imread(img_path)
        detections, _ = yolo.detect(image, conf_thres=0.001, iou_thres=0.65)

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            width = x2 - x1
            height = y2 - y1
            result = {
                "image_id": img_id,
                "category_id": cat_ids[int(cls)],
                "bbox": [float(x1), float(y1), float(width), float(height)],
                "score": float(conf)
            }
            results.append(result)

        if save_viz:
            viz_image = yolo.visualize_detections(image, detections)
            viz_output_path = os.path.join(viz_output_dir, img_info["file_name"])
            cv2.imwrite(viz_output_path, viz_image)

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 models using the COCO mAP metric")
    parser.add_argument("--model_path", required=False, default="yolov8n.onnx", help="Path to the ONNX model file")
    parser.add_argument("--val_dir", required=False, default="val2017",  help="Path to the COCO val2017 folder containing validation images")
    parser.add_argument("--annotation_file",  required=False, default="annotations/instances_val2017.json", help="Path to the instances_val2017.json annotation file")
    parser.add_argument("--save_viz", action='store_true', default=False, required=False, help="save visualizations if argument is passed")
    args = parser.parse_args()

    main(args.model_path, args.val_dir, args.annotation_file, args.save_viz)

