from flask import Flask, render_template, request, jsonify
import base64
import cv2
import numpy as np
import detection
from nms import COCO_NAMES

app = Flask(__name__)

yolo_models = {
    'n': 'yolov8n.onnx',
    's': 'yolov8s.onnx',
    'm': 'yolov8m.onnx',
    'l': 'yolov8l.onnx',
    'x': 'yolov8x.onnx',
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    json_data = request.get_json()
    model_type = json_data.get('model')
    model_path = yolo_models.get(model_type, 'yolov8s.onnx')
    yolo = detection.YOLOv8ONNX(model_path)

    image_data = json_data.get('image_data')
    image_data = base64.b64decode(image_data.split(',')[1])
    image = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), flags=cv2.IMREAD_COLOR)

    detections, preprocessed_image_shape = yolo.detect(image)
    detections_json = []
    for det in detections:
        if det.shape[0] >= 6:
            x1, y1, x2, y2, conf, cls = det
            detections_json.append({'x1': x1.item(), 'y1': y1.item(), 'x2': x2.item(), 'y2': y2.item(), 'conf': conf.item(), 'cls': str(COCO_NAMES[int(cls.item())])})

    return jsonify(detections_json)

if __name__ == '__main__':
    app.run(debug=True)
