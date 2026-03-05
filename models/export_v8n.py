from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # or yolov8s.pt
model.export(format="tflite", imgsz=640, half=True)  # fp16
# or int8: model.export(format="tflite", imgsz=640, int8=True)