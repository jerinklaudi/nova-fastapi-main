"""
NOVA - OCR Module (PaddleOCR Version)
=====================================
Optimized for Offline-First Assistive Vision
Architecture: PP-OCRv4 Mobile
Model Size: ~15–20 MB
"""

import os

# -----------------------------------------------------------------------------
# Paddle/PaddleOCR runtime flags (must be set BEFORE importing paddle/paddleocr)
# -----------------------------------------------------------------------------
# Avoid online checks where supported
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"  # 🔒 Offline guarantee

# Work around Paddle 3.x CPU executor issues seen on Windows (PIR/oneDNN errors)
# These flags are safe no-ops if unsupported in your installed Paddle build.
os.environ.setdefault("FLAGS_use_mkldnn", "0")  # disable oneDNN/MKLDNN kernels
os.environ.setdefault("FLAGS_enable_pir_api", "0")  # disable PIR API
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")  # disable PIR in executor
os.environ.setdefault("FLAGS_new_executor", "0")  # prefer legacy executor when available


import cv2
import numpy as np
import time
import re
from wordfreq import zipf_frequency

# PaddleOCR (paddlepaddle) and torch can conflict on Windows due to DLL/runtime clashes.
# Default to EasyOCR to avoid importing paddle/paddlex at runtime.
# You can force PaddleOCR by setting: NOVA_OCR_BACKEND=paddleocr
OCR_BACKEND = os.environ.get("NOVA_OCR_BACKEND", "easyocr").strip().lower()
PaddleOCR = None  # type: ignore
easyocr = None  # type: ignore

if OCR_BACKEND == "paddleocr":
    from paddleocr import PaddleOCR  # type: ignore
else:
    import easyocr  # type: ignore


class PaddleOCRSystem:
    def __init__(self, lang="en"):
        print("=" * 60)
        print("NOVA OCR Module - Initializing (PaddleOCR v4 Mobile)")
        print("=" * 60)

        self.lang = lang
        if OCR_BACKEND == "paddleocr":
            # Keep it CPU-only.
            self.reader = PaddleOCR(lang=lang, use_angle_cls=True, use_gpu=False, show_log=False)  # type: ignore[misc]
            backend_msg = "PaddleOCR"
        else:
            # EasyOCR expects language codes like ['en'].
            self.reader = easyocr.Reader([lang], gpu=False)  # type: ignore[name-defined]
            backend_msg = "EasyOCR"

        # Avoid unicode symbols (✓) to prevent Windows console encoding crashes
        print("OK: Device: CPU")
        print(f"OK: Language: {lang}")
        print(f"OK: Backend: {backend_msg}")
        print("OK: Offline after first successful model download")
        print("=" * 60)

    # --------------------------------------------------
    # OCR CORE
    # --------------------------------------------------
    def recognize_text(self, image_path=None, frame=None, visualize=True):
        if image_path:
            img = cv2.imread(image_path)
        else:
            img = frame

        if img is None:
            raise ValueError("Invalid image input")

        start = time.perf_counter()

        if OCR_BACKEND == "paddleocr":
            # Prefer PaddleOCR 2.x API. Keep a fallback to v3 predict() if needed.
            try:
                raw_results = self.reader.ocr(img, cls=True)
            except TypeError:
                raw_results = self.reader.ocr(img)
            except AttributeError:
                raw_results = self.reader.predict(img)
        else:
            # EasyOCR returns: [ (bbox, text, conf), ... ] with bbox as 4 points
            raw_results = self.reader.readtext(img)  # type: ignore[union-attr]
        inference_time = time.perf_counter() - start

        text_blocks = []
        full_text = []
        annotated_img = img.copy()

        # Normalize raw_results into an iterable of lines:
        # - PaddleOCR: line = [bbox, (text, conf)]
        # - EasyOCR:   line = (bbox, text, conf)
        lines = []
        if raw_results:
            if OCR_BACKEND == "paddleocr":
                if isinstance(raw_results, list) and len(raw_results) == 1 and isinstance(raw_results[0], list):
                    lines = raw_results[0]
                elif isinstance(raw_results, list):
                    lines = raw_results
            else:
                lines = raw_results

        if lines:
            for line in lines:
                if OCR_BACKEND == "paddleocr":
                    bbox = line[0]
                    text = line[1][0]
                    confidence = float(line[1][1])
                else:
                    bbox = line[0]
                    text = line[1]
                    confidence = float(line[2])

                text_blocks.append({
                    "text": text,
                    "confidence": confidence,
                    "bbox": bbox
                })
                full_text.append(text)

                if visualize:
                    pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_img, [pts], True, (0, 255, 0), 2)
                    cv2.putText(
                        annotated_img,
                        text,
                        (int(bbox[0][0]), int(bbox[0][1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1
                    )

        return {
            "text_blocks": text_blocks,
            "full_text": " ".join(full_text),
            "inference_time": inference_time,
            "annotated_image": annotated_img,
            "num_blocks": len(text_blocks),
        }

    # --------------------------------------------------
    # TEXT CLEANING (FOR TTS SAFETY)
    # --------------------------------------------------
    def basic_cleanup(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9.,;:'\"!? ]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def remove_gibberish_words(self, text, threshold=2.5):
        words = text.split()
        clean_words = [
            w for w in words if zipf_frequency(w, "en") >= threshold
        ]
        return " ".join(clean_words)

    def clean_ocr_text(self, text):
        if not text:
            return ""
        text = self.basic_cleanup(text)
        text = self.remove_gibberish_words(text)
        return text

    # --------------------------------------------------
    # VIDEO MODE (CAPTURE ON DEMAND)
    # --------------------------------------------------
    def process_video(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        print("\nControls:")
        print("  c - Capture & OCR")
        print("  q - Quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("NOVA - PaddleOCR (Offline)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            elif key == ord("c"):
                print("\n[OCR] Capturing frame...")
                result = self.recognize_text(frame=frame, visualize=False)
                clean = self.clean_ocr_text(result["full_text"])

                print(f"[RAW]    {result['full_text']}")
                print(f"[CLEAN]  {clean}")
                print(f"[TIME]   {result['inference_time']:.2f}s")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ocr = PaddleOCRSystem()
    ocr.process_video()
