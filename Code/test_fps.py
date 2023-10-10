import argparse
import os
import cv2
from detection import YOLOv8ONNX
from tqdm import tqdm
import onnxruntime as ort
import time
import random


def main(model_path, val_dir, num_test_images):
    device = ort.get_device()
    print("using device:", device)
    yolo = YOLOv8ONNX(model_path, device=device)

    image_files = [f for f in os.listdir(val_dir) if f.endswith('.jpg') or f.endswith('.png')]
    total_time = 0

    for _ in tqdm(range(num_test_images)):
        img_file = random.choice(image_files)
        img_path = os.path.join(val_dir, img_file)
        image = cv2.imread(img_path)

        start_time = time.time()
        yolo.detect(image, conf_thres=0.25, iou_thres=0.45)
        total_time += time.time() - start_time

    fps = num_test_images / total_time
    print(f"FPS: {fps:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 models' FPS on the validation dataset")
    parser.add_argument("--model_path", required=False, default="yolov8n.onnx", help="Path to the ONNX model file")
    parser.add_argument("--val_dir", required=False, default="val2017",  help="Path to the COCO val2017 folder containing validation images")
    parser.add_argument("--num_test_images", required=False, default=600,  help="Performs test on this number of images")
    args = parser.parse_args()

    main(args.model_path, args.val_dir, args.num_test_images)
