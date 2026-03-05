# test_yolo_tflite_debug_fixed.py

import numpy as np
import cv2
import time
import tensorflow.lite as tflite

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
MODEL_PATH = r"C:\Users\jerin\Desktop\Jerin\SRM\Semesters\Semester 8\Major Project\NOVA\nova-fastapi-main\models\yolov5s-fp16.tflite"
IMG_PATH   = r"image.png"
OUTPUT_PATH = "output_detections_debug.jpg"

CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.45

# Full COCO 80 class names (YOLOv5 standard)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# ────────────────────────────────────────────────
# NMS using OpenCV (more stable)
# ────────────────────────────────────────────────
def non_max_suppression(boxes, scores, classes, conf_thres=0.25, iou_thres=0.45):
    if len(boxes) == 0:
        return np.empty((0, 6))

    # Filter low confidence first
    mask = scores > conf_thres
    boxes   = boxes[mask]
    scores  = scores[mask]
    classes = classes[mask]

    if len(boxes) == 0:
        return np.empty((0, 6))

    # Use OpenCV NMSBoxes
    indices = cv2.dnn.NMSBoxes(
        boxes.astype(np.float32).tolist(),
        scores.tolist(),
        conf_thres,
        iou_thres
    )

    if len(indices) > 0:
        indices = indices.flatten()
        return np.column_stack((boxes[indices], scores[indices], classes[indices]))
    return np.empty((0, 6))

# ────────────────────────────────────────────────
print("=== YOLOv5 TFLite - Debug Mode (Fixed) ===")
print("Image:", IMG_PATH)

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_shape = input_details['shape'][1:3]  # [640, 640]

orig_img = cv2.imread(IMG_PATH)
if orig_img is None:
    print("Cannot load image")
    exit(1)

orig_h, orig_w = orig_img.shape[:2]

# Preprocess
img = cv2.resize(orig_img, (input_shape[1], input_shape[0]))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
img = np.expand_dims(img, 0)

# Inference
t0 = time.time()
interpreter.set_tensor(input_details['index'], img)
interpreter.invoke()
raw_output = interpreter.get_tensor(output_details['index'])[0]  # [25200, 85]
print(f"Inference: {(time.time()-t0)*1000:.1f} ms   Output shape: {raw_output.shape}")

# ── Extract ───────────────────────────────────────
cxcywh   = raw_output[:, :4]
obj_conf = raw_output[:, 4]
cls_pred = raw_output[:, 5:]

# Combined confidence = obj_conf * max class prob
scores = obj_conf * cls_pred.max(axis=1)
class_ids = cls_pred.argmax(axis=1)

# Boxes cxcywh → xyxy
x1 = cxcywh[:, 0] - cxcywh[:, 2] / 2
y1 = cxcywh[:, 1] - cxcywh[:, 3] / 2
x2 = cxcywh[:, 0] + cxcywh[:, 2] / 2
y2 = cxcywh[:, 1] + cxcywh[:, 3] / 2
boxes = np.column_stack((x1, y1, x2, y2))

# ── Scale decision ─────────────────────────────────
max_coord = boxes.max()
if max_coord > 2.0:
    print("→ Detected PIXEL-scale coordinates (no extra scaling)")
    scale_x = scale_y = 1.0
else:
    print(f"→ Detected NORMALIZED coordinates (max coord={max_coord:.3f}) → scaling")
    scale_x = orig_w / input_shape[1]
    scale_y = orig_h / input_shape[0]

boxes[:, [0, 2]] *= scale_x
boxes[:, [1, 3]] *= scale_y

# Clip to image bounds
boxes = np.clip(boxes, [0, 0, 0, 0], [orig_w, orig_h, orig_w, orig_h])

# NMS
keep = non_max_suppression(boxes, scores, class_ids, CONF_THRESHOLD, IOU_THRESHOLD)

print(f"\nAfter NMS (conf >= {CONF_THRESHOLD}): {len(keep)} detections")

# ── Print kept detections for debug ───────────────
if len(keep) > 0:
    print("\nKept detections (x1,y1,x2,y2 | conf | class):")
    for i, det in enumerate(keep):
        x1, y1, x2, y2, conf, cls_id = det
        label = COCO_CLASSES[int(cls_id)] if int(cls_id) < len(COCO_CLASSES) else f"cls_{int(cls_id)}"
        print(f"  #{i+1}: [{int(x1):4d}, {int(y1):4d}, {int(x2):4d}, {int(y2):4d}]  conf={conf:.3f}  {label}")
else:
    print("No detections kept. Try lowering CONF_THRESHOLD to 0.15 or 0.10")

# ── Draw ──────────────────────────────────────────
color = (0, 255, 0)
thickness = 4   # thicker lines
font_scale = 1.0

for det in keep:
    x1, y1, x2, y2, conf, cls_id = map(int, det)
    label = COCO_CLASSES[int(cls_id)] if int(cls_id) < len(COCO_CLASSES) else f"cls_{int(cls_id)}"

    # Box
    cv2.rectangle(orig_img, (x1, y1), (x2, y2), color, thickness)

    # Label background + text
    txt = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    cv2.rectangle(orig_img, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
    cv2.putText(orig_img, txt, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)

cv2.imwrite(OUTPUT_PATH, orig_img)
print(f"\nSaved result: {OUTPUT_PATH}")

cv2.imshow("YOLO Detections", orig_img)
cv2.waitKey(0)
cv2.destroyAllWindows()