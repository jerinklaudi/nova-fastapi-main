import numpy as np
import cv2
import time
import tensorflow.lite as tflite

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────
MODEL_PATH = r"C:\Users\jerin\Desktop\Jerin\SRM\Semesters\Semester 8\Major Project\NOVA\nova-fastapi-main\models\yolov8n_float16.tflite"

CONF_THRESHOLD = 0.25

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
# Load model
# ────────────────────────────────────────────────
print("Loading YOLOv8 TFLite model...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_h, input_w = input_details['shape'][1], input_details['shape'][2]
output_shape = output_details['shape']

print(f"Model loaded. Input shape:  {input_details['shape']}")
print(f"             Output shape: {output_shape}")
# Determine output format
# [1, 300, 6] → NMS-included export: each row is [x1, y1, x2, y2, score, class_id]

# ────────────────────────────────────────────────
# Debug: inspect raw output once to confirm coordinate range
# ────────────────────────────────────────────────
DEBUG_FIRST_FRAME = True

# ────────────────────────────────────────────────
# Webcam
# ────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

print("Webcam opened. Press 'q' to quit.")
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    orig_h, orig_w = frame.shape[:2]

    # ── Pre-process ─────────────────────────────────────────
    img_resized = cv2.resize(frame, (input_w, input_h))
    img_input   = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_input   = np.expand_dims(img_input, axis=0)  # [1, 640, 640, 3]

    # ── Inference ───────────────────────────────────────────
    interpreter.set_tensor(input_details['index'], img_input)
    interpreter.invoke()

    # Output: [1, 300, 6]  → [300, 6]
    raw_output = interpreter.get_tensor(output_details['index'])[0]  # [300, 6]

    if DEBUG_FIRST_FRAME:
        print("\n── DEBUG: First 5 raw detections ──")
        print(raw_output[:5])
        print(f"Score range in this frame: min={raw_output[:, 4].min():.3f}  max={raw_output[:, 4].max():.3f}")
        print(f"Coord range: min={raw_output[:, :4].min():.3f}  max={raw_output[:, :4].max():.3f}")
        DEBUG_FIRST_FRAME = False

    # ── Parse [x1, y1, x2, y2, score, class_id] ────────────
    scores   = raw_output[:, 4]
    mask     = scores > CONF_THRESHOLD
    filtered = raw_output[mask]

    detections = []
    for row in filtered:
        x1_raw, y1_raw, x2_raw, y2_raw = row[0], row[1], row[2], row[3]
        score  = float(row[4])
        cls_id = int(row[5])

        # Ultralytics NMS-embedded TFLite exports use NORMALIZED coords [0–1]
        # If coords are > 1.5 they are already in pixel space (640-scale), handle both:
        if x2_raw <= 1.5:
            # Normalized → scale to frame size
            x1 = int(np.clip(x1_raw * orig_w, 0, orig_w))
            y1 = int(np.clip(y1_raw * orig_h, 0, orig_h))
            x2 = int(np.clip(x2_raw * orig_w, 0, orig_w))
            y2 = int(np.clip(y2_raw * orig_h, 0, orig_h))
        else:
            # Already in 640-pixel space → scale to frame size
            x1 = int(np.clip(x1_raw * orig_w / input_w, 0, orig_w))
            y1 = int(np.clip(y1_raw * orig_h / input_h, 0, orig_h))
            x2 = int(np.clip(x2_raw * orig_w / input_w, 0, orig_w))
            y2 = int(np.clip(y2_raw * orig_h / input_h, 0, orig_h))

        # Sanity check — skip degenerate boxes
        if x2 <= x1 or y2 <= y1:
            continue

        detections.append((x1, y1, x2, y2, score, cls_id))

    print(f"Detected {len(detections)} objects")

    # ── Draw ────────────────────────────────────────────────
    for (x1, y1, x2, y2, score, cls_id) in detections:
        label = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"cls{cls_id}"
        text  = f"{label} {score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(frame, text, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    # ── FPS ─────────────────────────────────────────────────
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time + 1e-6)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("YOLOv8n TFLite Live Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")