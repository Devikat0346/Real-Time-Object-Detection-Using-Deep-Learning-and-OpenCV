import cv2, os
import numpy as np
import onnxruntime as ort
import nms

class YOLOv8ONNX:
    def __init__(self, model_path, device="cpu"):
        self.model_path = model_path
        self.device = device
        if self.device =="cpu":
            self.model = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        else:
            self.model = ort.InferenceSession(self.model_path, providers=["CUDAExecutionProvider"])

    def preprocess(self, img, size=640):
        width = size
        height = size
        I = img
        w, h, _ = I.shape
        padded_img = np.ones((height, width, 3), dtype=np.float32) * 114
        ratio = min(height / h, width / w)
        resized_h = int(round(h * ratio))
        resized_w = int(round(w * ratio))
        img_resized = cv2.resize(I, (resized_h, resized_w), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        dh = (height - resized_h)/2
        dw = (width - resized_w)/2
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        padded_img[left: resized_w + left, top: resized_h+ top] = img_resized
        padded_img = padded_img.transpose(2,0,1)[::-1] #BGR2RGB
        padded_img = padded_img / 255
        padded_img = np.float32(np.expand_dims(padded_img,axis=0))
        return padded_img


    def postprocess(self, detections, orig_img_shape, preprocessed_img_shape, conf_thres=0.1, iou_thres=0.65):
        dets = nms.non_max_suppression(detections, conf_thres, iou_thres, classes=None, agnostic=False, max_det=100)
        dets[0][:, :4] = self.scale_boxes(orig_img_shape, dets[0][:, :4], preprocessed_img_shape).round()
        return dets[0]

    def scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None):
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
        boxes[..., [0, 2]] = (boxes[..., [0, 2]] - pad[0]) / gain
        boxes[..., [1, 3]] = (boxes[..., [1, 3]] - pad[1]) / gain
        boxes = self.clip_boxes(boxes, img0_shape)
        return boxes

    def detect(self, img, conf_thres=0.35, iou_thres=0.65):
        img_tensor = self.preprocess(img)

        outputs = self.model.run(None, {'images': img_tensor})
        detections = outputs[0]

        orig_img_shape = img.shape[:2]
        preprocessed_img_shape = img_tensor.shape[2:]
        results = self.postprocess(detections, preprocessed_img_shape,orig_img_shape, conf_thres, iou_thres)
        return results, preprocessed_img_shape

    def clip_boxes(self, boxes, shape):
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
        return boxes

    def visualize_detections(self, image, detections, class_names = nms.COCO_NAMES, colors = nms._COLORS, conf_thres=0.25):
        img = image.copy()
        for det in detections:
            if det.shape[0] < 6:  # Skip if detection does not have the correct shape
                continue
            xyxy, conf, cls = det[:4], det[4], int(det[5])
            if conf > conf_thres:
                label = f"{class_names[cls]} {conf:.2f}"
                color = [int(c * 255) for c in colors[cls]]
                img = self.draw_bbox(img, xyxy, label, color)
        return img


    def draw_bbox(self, img, bbox, label, color):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label, font, 0.5, 1)[0]
        cv2.rectangle(img, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 4), font, 0.5, (255, 255, 255), 1)
        return img

if __name__ == "__main__":
    model_path = "yolov8x.onnx"
    yolo = YOLOv8ONNX(model_path)

    img_path = "detection_test.png"
    img = cv2.imread(img_path)

    detections, preprocessed_image_shape = yolo.detect(img)
    vis_image = yolo.visualize_detections(img, np.array(detections), nms.COCO_NAMES, nms._COLORS)
    output_folder = "viz_output"
    os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist
    # Save the image
    output_path = os.path.join(output_folder, "detections.jpeg")
    cv2.imwrite(output_path, vis_image)
