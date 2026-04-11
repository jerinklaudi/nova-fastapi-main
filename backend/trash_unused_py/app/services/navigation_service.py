import cv2
import numpy as np
import torch
import time
import collections
import sys
from pathlib import Path
from fastapi import HTTPException

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
YOLO_MODEL_PATH = BASE_DIR / "models" / "yolov5s-fp16.tflite"
MIDAS_MODEL_PATH = BASE_DIR / "models" / "midas_v21_small_256.pt"

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    raise RuntimeError("Install tflite_runtime or tensorflow.")

CONF_THRESHOLD = 0.55
NMS_THRESHOLD = 0.45
MIN_AREA_RATIO = 0.01
DEPTH_PERCENTILE = 65
SMOOTHING_BUFFER = 5
DEPTH_INTERVAL = 4

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]


class AssistiveNavigator:
    def __init__(self):
        self.yolo_interpreter = None
        self.yolo_input_details = None
        self.yolo_output_details = None
        self.midas_model = None
        self.midas_transform = None
        self.depth_map = None
        self.frame_count = 0
        self.nav_buffer = collections.deque(maxlen=SMOOTHING_BUFFER)
        self.fps = 0.0
        self.prev_time = time.time()
        self.device = torch.device("cpu")

        # --- Navigation geometry thresholds ---
        self.lower_fraction    = 0.52   # fraction of frame height used as walking region
        self.near_percentile   = 78     # percentile for dynamic near_threshold
        self.stop_threshold    = 0.65   # risk above this → STOP
        self.caution_threshold = 0.40   # risk above this → CAUTION
        self.side_bias         = 0.22   # delta needed to prefer left/right turn
        self.ema_alpha         = 0.65   # EMA weight on previous risk (smoothing)

        # High-risk COCO classes (semantic boost multiplier ×1.6)
        self.high_risk_classes = {
            "person", "bicycle", "car", "motorcycle", "bus", "truck",
            "dog", "cat", "horse", "cow", "traffic light", "stop sign",
            "fire hydrant", "skateboard", "stroller"
        }

        # EMA state (scalar, no deque) — initialised to 0 on first frame
        self.prev_left_risk   = 0.0
        self.prev_center_risk = 0.0
        self.prev_right_risk  = 0.0

    def load_yolo_model(self):
        print(f"\n[DEBUG] Loading YOLO from: {YOLO_MODEL_PATH}")
        print(f"[DEBUG] File exists: {YOLO_MODEL_PATH.exists()}")
        
        if not YOLO_MODEL_PATH.exists():
            raise FileNotFoundError(f"YOLO model not found at: {YOLO_MODEL_PATH}")
            
        try:
            interpreter = tflite.Interpreter(model_path=str(YOLO_MODEL_PATH))
            interpreter.allocate_tensors()
            self.yolo_interpreter = interpreter
            self.yolo_input_details = interpreter.get_input_details()
            self.yolo_output_details = interpreter.get_output_details()
            print(f"[INFO] YOLO loaded: {YOLO_MODEL_PATH}")
            print(f"[DEBUG] YOLO input details:  {self.yolo_input_details}")
            print(f"[DEBUG] YOLO output details: {self.yolo_output_details}")
            print(f"[DEBUG] Input dtype:  {self.yolo_input_details[0]['dtype']}")
            print(f"[DEBUG] Output dtype: {self.yolo_output_details[0]['dtype']}")
            print(f"[DEBUG] Input shape:  {self.yolo_input_details[0]['shape']}")
            print(f"[DEBUG] Output shape: {self.yolo_output_details[0]['shape']}")
        except Exception as e:
            print(f"[ERROR] Failed to load YOLO model: {e}")
            raise HTTPException(status_code=500, detail="Navigation model not available")

    def load_midas_model(self):
        print(f"\n[DEBUG] Attempting to load MiDaS model (MiDaS_small) via torch.hub...")
        print(f"[DEBUG] Loading MiDaS from: {MIDAS_MODEL_PATH}")
        print(f"[DEBUG] File exists: {MIDAS_MODEL_PATH.exists()}")
        
        if not MIDAS_MODEL_PATH.exists():
            raise FileNotFoundError(f"MiDaS model not found at: {MIDAS_MODEL_PATH}")
            
        try:
            print("[INFO] Loading MiDaS model...")

            self.midas_model = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
            )
            print(f"[DEBUG] MiDaS architecture loaded successfully.")

            print(f"[DEBUG] Loading local weights from: {MIDAS_MODEL_PATH}")
            state_dict = torch.load(str(MIDAS_MODEL_PATH), map_location=self.device)
            self.midas_model.load_state_dict(state_dict)
            print(f"[DEBUG] Local weights loaded successfully.")

            self.midas_model.to(self.device)
            self.midas_model.eval()
            print(f"[DEBUG] MiDaS set to eval mode on device: {self.device}")

            transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms", trust_repo=True
            )
            self.midas_transform = transforms.small_transform
            print("[INFO] MiDaS loaded successfully (local weights).")

        except Exception as e:
            print(f"[ERROR] Failed to load MiDaS model: {e}")
            raise HTTPException(status_code=500, detail="Navigation model not available")

    def run_yolo(self, frame):
        input_size = 640
        img = cv2.resize(frame, (input_size, input_size))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_norm, axis=0)

        input_detail = self.yolo_input_details[0]
        if input_detail['dtype'] == np.uint8:
            scale, zero_point = input_detail['quantization']
            img_batch = (img_batch / scale + zero_point).astype(np.uint8)
        elif input_detail['dtype'] == np.int8:
            scale, zero_point = input_detail['quantization']
            img_batch = (img_batch / scale + zero_point).astype(np.int8)

        self.yolo_interpreter.set_tensor(input_detail['index'], img_batch)
        self.yolo_interpreter.invoke()

        print("\n[DEBUG] All output tensors:")
        for i, detail in enumerate(self.yolo_output_details):
            name = detail.get('name', f'output_{i}')
            shape = detail['shape']
            tensor = self.yolo_interpreter.get_tensor(detail['index'])
            print(f"  {i}) {name:20} shape = {shape}   dtype = {tensor.dtype}   min/max = {tensor.min():.3f} / {tensor.max():.3f}")

        output_detail = self.yolo_output_details[0]
        output = self.yolo_interpreter.get_tensor(output_detail['index'])

        if output_detail['dtype'] in (np.uint8, np.int8):
            scale, zero_point = output_detail['quantization']
            output = (output.astype(np.float32) - zero_point) * scale

        print(f"[DEBUG][YOLO] Raw output shape: {output.shape} | min={output.min():.4f} max={output.max():.4f}")
        return output

    def run_midas(self, frame):
        print(f"[DEBUG][MiDaS] Running depth estimation on frame of shape {frame.shape}")
        img = cv2.resize(frame, (256, 256))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_rgb = (img_rgb - mean) / std

        img_rgb = np.transpose(img_rgb, (2, 0, 1))
        input_batch = torch.from_numpy(img_rgb).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            prediction = self.midas_model(input_batch)

        depth = prediction.squeeze().cpu().numpy()
        print(f"[DEBUG][MiDaS] Raw depth output shape: {depth.shape} | min={depth.min():.4f} max={depth.max():.4f}")

        depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))

        d_min = depth.min()
        d_max = depth.max()
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)
            print(f"[DEBUG][MiDaS] WARNING: Depth map is flat (min==max), returning zeros.")

        print(f"[DEBUG][MiDaS] Normalized depth map | min={depth.min():.4f} max={depth.max():.4f} mean={depth.mean():.4f}")
        return depth

    def decode_yolo_output(self, output, frame_shape):
        orig_h, orig_w = frame_shape[:2]
        input_size = 640
        predictions = output[0]

        print(f"\n[DEBUG][DECODE] Total raw predictions (anchors): {len(predictions)}")
        print(f"[DEBUG][DECODE] Frame size: {orig_w}x{orig_h} | Frame area: {orig_w * orig_h}")

        boxes = []
        confidences = []
        class_ids = []

        rejected_low_obj_conf = 0
        rejected_low_final_conf = 0

        for i, pred in enumerate(predictions):
            if i < 3:
                print("RAW BOX SAMPLE:", pred[:4])
            obj_conf = float(pred[4])
            if obj_conf < CONF_THRESHOLD:
                rejected_low_obj_conf += 1
                continue
            class_scores = pred[5:]
            class_id = int(np.argmax(class_scores))
            class_score = float(class_scores[class_id])
            final_conf = obj_conf * class_score
            if final_conf < CONF_THRESHOLD:
                rejected_low_final_conf += 1
                continue

            print(f"RAW BOX NORM: cx={pred[0]:.4f}, cy={pred[1]:.4f}, w={pred[2]:.4f}, h={pred[3]:.4f}")

            cx = float(pred[0]) * orig_w
            cy = float(pred[1]) * orig_h
            w  = float(pred[2]) * orig_w
            h  = float(pred[3]) * orig_h

            print(f"SCALED BOX: cx={cx:.1f}, cy={cy:.1f}, w={w:.1f}, h={h:.1f}")

            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            bw = int(w)
            bh = int(h)

            boxes.append([x1, y1, bw, bh])
            confidences.append(float(final_conf))
            class_ids.append(class_id)

        print(f"[DEBUG][DECODE] Rejected (obj_conf < {CONF_THRESHOLD}): {rejected_low_obj_conf}")
        print(f"[DEBUG][DECODE] Rejected (final_conf < {CONF_THRESHOLD}): {rejected_low_final_conf}")
        print(f"[DEBUG][DECODE] Passed confidence filter (pre-NMS): {len(boxes)}")

        if len(boxes) > 0:
            print(f"[DEBUG][DECODE] Pre-NMS detections:")
            for idx, (b, c, cid) in enumerate(zip(boxes, confidences, class_ids)):
                cname = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else "unknown"
                print(f"           [{idx}] class={cname}({cid}) conf={c:.4f} box=[x1={b[0]}, y1={b[1]}, w={b[2]}, h={b[3]}]")

        detections = []
        if len(boxes) == 0:
            print(f"[DEBUG][DECODE] No boxes to run NMS on. Returning empty detections.")
            return detections

        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
        if indices is None or len(indices) == 0:
            print(f"[DEBUG][DECODE] NMS returned no boxes. Returning empty detections.")
            return detections

        if isinstance(indices, np.ndarray):
            indices = indices.flatten().tolist()
        elif hasattr(indices, 'flatten'):
            indices = indices.flatten().tolist()

        print(f"[DEBUG][DECODE] After NMS: {len(indices)} boxes remain")

        frame_area = orig_w * orig_h
        min_area = MIN_AREA_RATIO * frame_area
        rejected_small = 0

        for i in indices:
            x1, y1, bw, bh = boxes[i]
            x2 = x1 + bw
            y2 = y1 + bh
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(orig_w - 1, x2)
            y2 = min(orig_h - 1, y2)
            area = (x2 - x1) * (y2 - y1)
            cname = COCO_CLASSES[class_ids[i]] if class_ids[i] < len(COCO_CLASSES) else "unknown"
            if area < min_area:
                print(f"[DEBUG][DECODE] Rejected '{cname}' — area {area} < min_area {min_area:.1f} ({MIN_AREA_RATIO*100}% of frame)")
                rejected_small += 1
                continue
            detections.append({
                "box": (x1, y1, x2, y2),
                "confidence": confidences[i],
                "class_id": class_ids[i],
                "class_name": cname
            })

        print(f"[DEBUG][DECODE] Rejected (too small): {rejected_small}")
        print(f"[DEBUG][DECODE] Final detections passed all filters: {len(detections)}")
        return detections

    def _max_free_run(self, mask):
        max_run = 0
        current = 0
        for row in mask:
            current = 0
            for val in row:
                if val:
                    current += 1
                    if current > max_run:
                        max_run = current
                else:
                    current = 0
        return max_run

    def compute_navigation(self, detections, depth_map, frame_shape):
        """Geometry-primary navigation decision.

        Primary signal: depth-map free-space estimation in the lower walking
        region.  YOLO detections act only as a secondary semantic boost.
        Returns a result dict (never a bare string).
        """
        if depth_map is None:
            return {
                "command": "INITIALIZING",
                "risks": {"left": 0.0, "center": 0.0, "right": 0.0},
                "free_ratios": {"left": 1.0, "center": 1.0, "right": 1.0},
                "near_threshold": 0.0,
                "center_occupancy": 0.0,
                "suggested_direction": None,
            }

        orig_h, orig_w = frame_shape[:2]
        frame_area = float(orig_w * orig_h)

        # ── 1. Extract lower walking region ──────────────────────────────────
        lower_start = int(orig_h * self.lower_fraction)
        lower_half  = depth_map[lower_start:, :]          # (lH, W) view
        lH, lW      = lower_half.shape

        # ── 2. Dynamic near threshold ─────────────────────────────────────────
        near_threshold = float(np.percentile(lower_half, self.near_percentile))

        # ── 3. Occupancy masks ────────────────────────────────────────────────
        occupied_mask = lower_half >= near_threshold      # True = occupied/close
        free_mask     = ~occupied_mask

        # ── 4. Split into 3 equal vertical zones (LEFT / CENTER / RIGHT) ─────
        z0 = lW // 3
        z1 = (2 * lW) // 3
        zone_slices = [
            (slice(None), slice(0,    z0)),   # LEFT
            (slice(None), slice(z0,   z1)),   # CENTER
            (slice(None), slice(z1, None)),   # RIGHT
        ]

        zone_risks      = []
        zone_free_ratios = []

        for s in zone_slices:
            lh_zone   = lower_half[s]
            fm_zone   = free_mask[s]
            total_px  = lh_zone.size          # guaranteed > 0
            free_px   = int(fm_zone.sum())
            free_ratio = free_px / total_px

            # max contiguous horizontal free run (vectorised)
            max_run   = self._max_free_run(fm_zone)
            zone_width = lh_zone.shape[1]
            max_free_width = max_run / zone_width if zone_width > 0 else 0.0

            mean_depth_zone = float(lh_zone.mean())

            # zone risk = weighted combination of occupancy signals
            z_risk = (
                0.70 * (1.0 - free_ratio) +
                0.20 * (1.0 - max_free_width) +
                0.10 * mean_depth_zone
            )
            zone_risks.append(float(np.clip(z_risk, 0.0, 1.0)))
            zone_free_ratios.append(free_ratio)

        left_risk_raw, center_risk_raw, right_risk_raw = zone_risks

        # ── 5. Semantic boost from YOLO detections ────────────────────────────
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            box_area = float((x2 - x1) * (y2 - y1))
            area_ratio = box_area / frame_area

            # Compute overlap fraction with lower_half region
            overlap_y1  = max(y1, lower_start)
            overlap_y2  = min(y2, orig_h)
            if overlap_y2 <= overlap_y1:
                continue
            overlap_h    = overlap_y2 - overlap_y1
            overlap_area = overlap_h * (x2 - x1)
            if overlap_area / max(box_area, 1.0) < 0.20:
                continue   # less than 20% of box in walking region → skip

            boost = det["confidence"] * area_ratio
            if det["class_name"] in self.high_risk_classes:
                boost *= 1.6

            # Assign boost to zone by box centre-x
            cx = (x1 + x2) / 2.0
            if cx < (orig_w / 3.0):
                left_risk_raw   = min(1.0, left_risk_raw   + boost)
            elif cx < (2.0 * orig_w / 3.0):
                center_risk_raw = min(1.0, center_risk_raw + boost)
            else:
                right_risk_raw  = min(1.0, right_risk_raw  + boost)

        # ── 6. EMA risk smoothing (scalar, no deque) ──────────────────────────
        a = self.ema_alpha
        left_risk   = a * self.prev_left_risk   + (1.0 - a) * left_risk_raw
        center_risk = a * self.prev_center_risk + (1.0 - a) * center_risk_raw
        right_risk  = a * self.prev_right_risk  + (1.0 - a) * right_risk_raw

        self.prev_left_risk   = left_risk
        self.prev_center_risk = center_risk
        self.prev_right_risk  = right_risk

        # ── 7. Decision logic ─────────────────────────────────────────────────
        suggested_direction = None

        if (left_risk > self.stop_threshold and
                center_risk > self.stop_threshold and
                right_risk  > self.stop_threshold):
            command = "STOP"

        elif center_risk > self.stop_threshold:
            command = "STOP"

        elif center_risk > self.caution_threshold:
            command = "CAUTION"
            # Suggest the less risky side
            suggested_direction = "LEFT" if left_risk <= right_risk else "RIGHT"

        elif left_risk > self.stop_threshold and right_risk > self.stop_threshold:
            command = "CAUTION"
            suggested_direction = None

        elif left_risk > right_risk + self.side_bias:
            command = "MOVE_RIGHT"

        elif right_risk > left_risk + self.side_bias:
            command = "MOVE_LEFT"

        else:
            command = "PATH_CLEAR"

        # Safety veto: never claim clear if centre is mostly occupied
        if command == "PATH_CLEAR" and zone_free_ratios[1] < 0.55:
            command = "CAUTION"
            suggested_direction = "LEFT" if left_risk <= right_risk else "RIGHT"

        center_occupancy = 1.0 - zone_free_ratios[1]

        print(f"[DEBUG][NAV] risks L={left_risk:.3f} C={center_risk:.3f} R={right_risk:.3f} "
              f"| free L={zone_free_ratios[0]:.2f} C={zone_free_ratios[1]:.2f} R={zone_free_ratios[2]:.2f} "
              f"| cmd={command}")

        return {
            "command": command,
            "risks": {
                "left":   round(left_risk,   4),
                "center": round(center_risk, 4),
                "right":  round(right_risk,  4),
            },
            "free_ratios": {
                "left":   round(zone_free_ratios[0], 4),
                "center": round(zone_free_ratios[1], 4),
                "right":  round(zone_free_ratios[2], 4),
            },
            "near_threshold":    round(near_threshold, 4),
            "center_occupancy":  round(center_occupancy, 4),
            "suggested_direction": suggested_direction,
        }

    def apply_temporal_smoothing(self, command):
        self.nav_buffer.append(command)
        counter = collections.Counter(self.nav_buffer)
        smoothed = counter.most_common(1)[0][0]
        print(f"[DEBUG][SMOOTH] Buffer: {list(self.nav_buffer)}")
        print(f"[DEBUG][SMOOTH] Vote counts: {dict(counter)}")
        print(f"[DEBUG][SMOOTH] Raw='{command}'  Smoothed='{smoothed}'")
        return smoothed

    def draw_overlay(self, frame, detections, depth_map, final_command):
        orig_h, orig_w = frame.shape[:2]
        zone_w = orig_w // 3

        if depth_map is not None:
            depth_colored = (depth_map * 255).astype(np.uint8)
            depth_heatmap = cv2.applyColorMap(depth_colored, cv2.COLORMAP_MAGMA)
            frame = cv2.addWeighted(frame, 0.65, depth_heatmap, 0.35, 0)

        cv2.line(frame, (zone_w, 0), (zone_w, orig_h), (255, 255, 0), 1)
        cv2.line(frame, (2 * zone_w, 0), (2 * zone_w, orig_h), (255, 255, 0), 1)

        cv2.putText(frame, "LEFT", (zone_w // 4, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(frame, "CENTER", (zone_w + zone_w // 4, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(frame, "RIGHT", (2 * zone_w + zone_w // 4, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            conf = det["confidence"]
            name = det["class_name"]
            depth_val = det.get("depth_value", -1)

            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{name} {conf:.2f}"
            if depth_val >= 0:
                label += f" d:{depth_val:.2f}"

            label_y = y1 - 8 if y1 > 20 else y1 + 16
            cv2.putText(frame, label, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cmd_colors = {
            "STOP": (0, 0, 255),
            "MOVE RIGHT": (0, 165, 255),
            "MOVE LEFT": (0, 165, 255),
            "PATH CLEAR": (0, 255, 0),
            "INITIALIZING": (200, 200, 200)
        }
        cmd_color = cmd_colors.get(final_command, (255, 255, 255))

        cv2.rectangle(frame, (0, orig_h - 50), (orig_w, orig_h), (0, 0, 0), -1)
        cv2.putText(frame, f"NAV: {final_command}", (10, orig_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, cmd_color, 2)

        cv2.putText(frame, f"FPS: {self.fps:.1f}", (orig_w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def run(self):
        self.load_yolo_model()
        self.load_midas_model()

        #print(f"\n[DEBUG][RUN] Opening video stream: http://192.168.1.6:8080/video")
        print("\n[INFO] Using Phone Link connected phone camera (index 1)...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("[ERROR] Cannot open phone camera at index 1.")
            print("    Double-check Phone Link is connected and camera enabled.")
            print("    Try index 0 (laptop) to confirm, or restart Phone Link/PC.")
            raise RuntimeError("Cannot open phone camera at index 1.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"[DEBUG][RUN] Requested capture resolution: 640x480")
        print("[INFO] Starting navigation loop. Press 'q' to quit.")


        ret, test_frame = cap.read()
        if ret:
            print(f"Test frame from phone: shape {test_frame.shape}, should be ~ (720,1280,3) or resized")
            cv2.imshow("Phone Camera Test (press any key to continue)", test_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Failed to read test frame — try different backend above.")


        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to read frame.")
                break

            if frame.shape[0] > frame.shape[1]:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                print(f"[DEBUG][RUN] Frame rotated (portraitlandscape). New shape: {frame.shape}")

            frame = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_AREA)
            self.frame_count += 1
            current_time = time.time()
            elapsed = current_time - self.prev_time
            if elapsed > 0:
                self.fps = 1.0 / elapsed
            self.prev_time = current_time

            print(f"\n{'#'*60}")
            print(f"[DEBUG][RUN] ===== FRAME {self.frame_count} | FPS={self.fps:.1f} =====")
            print(f"[DEBUG][RUN] Frame shape after resize: {frame.shape}")

            yolo_output = self.run_yolo(frame)
            detections = self.decode_yolo_output(yolo_output, frame.shape)
            print(f"[DEBUG][RUN] Detections after decode: {len(detections)}")

            run_depth = (self.frame_count % DEPTH_INTERVAL == 0 or self.depth_map is None)
            print(f"[DEBUG][RUN] Run MiDaS this frame: {run_depth} (frame_count={self.frame_count}, interval={DEPTH_INTERVAL})")
            if run_depth:
                self.depth_map = self.run_midas(frame)
            else:
                print(f"[DEBUG][RUN] Using cached depth map from previous frame.")

            nav_result = self.compute_navigation(detections, self.depth_map, frame.shape)
            raw_command = nav_result["command"]
            final_command = self.apply_temporal_smoothing(raw_command)
            print(f"[DEBUG][RUN] FINAL COMMAND  {final_command}")

            display_frame = self.draw_overlay(frame.copy(), detections, self.depth_map, final_command)

            cv2.imshow("Assistive Navigation", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[DEBUG][RUN] 'q' pressed — exiting loop.")
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Navigation system stopped.")


if __name__ == "__main__":
    navigator = AssistiveNavigator()
    navigator.run()