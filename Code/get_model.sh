#!/bin/sh
mkdir output 
cd output
git clone https://github.com/ultralytics/ultralytics.git
git checkout 0cb87f7dd340a2611148fbf2a0af59b544bd7b1b
cd ultralytics

## CREATING VIRTUAL ENV, INSTALLING PACKAGES, (IGNORE ERROR MESSAGES)
virtualenv yolov8app -p python3.7
source yolov8app/bin/activate
python -m pip install onnxruntime==1.12.0
python -m pip install onnx==1.12.0
python -m pip install onnxsim==0.4.13
python -m pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0
python -m pip install -r requirements.txt 
python -m pip install ultralytics

python -m pip list |grep onnx
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt

export PYTHONPATH=
yolo task=detect mode=export model=yolov8n.pt imgsz=640 format=onnx opset=12 simplify=True
yolo task=detect mode=export model=yolov8s.pt imgsz=640 format=onnx opset=12 simplify=True
yolo task=detect mode=export model=yolov8m.pt imgsz=640 format=onnx opset=12 simplify=True
yolo task=detect mode=export model=yolov8l.pt imgsz=640 format=onnx opset=12 simplify=True
yolo task=detect mode=export model=yolov8x.pt imgsz=640 format=onnx opset=12 simplify=True
deactivate