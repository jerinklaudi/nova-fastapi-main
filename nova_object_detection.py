"""
NOVA - Object Detection Module (Offline-First)
================================================
B.Tech Research Project - Assistive Vision System

WHY THIS MODEL:
- YOLOv5n (nano): Smallest YOLO variant at ~14 MB
- Trained on COCO dataset (80 classes including person, chair, bottle, etc.)
- Optimized for CPU inference with minimal latency
- Fully offline after initial model download
- No cloud APIs or internet dependency

PERFORMANCE TARGET:
- CPU-only inference: 15-25 FPS on modern laptop
- Inference time: ~40-70 ms per frame
- Model size: 14 MB (well under 400 MB limit)

ACADEMIC JUSTIFICATION:
- YOLOv5 is widely cited in assistive technology research
- Single-stage detector provides real-time performance
- Pre-trained weights ensure reproducibility
"""

import torch
import cv2
import time
import os
import numpy as np
from pathlib import Path


class OfflineObjectDetector:
    """
    Offline-first object detection using YOLOv5 nano.
    No internet required after initial setup.
    """
    
    def __init__(self, model_name='yolov5n', confidence_threshold=0.4):
        """
        Initialize detector with lightweight YOLOv5 nano model.
        
        Args:
            model_name: YOLOv5 variant (yolov5n = nano, smallest)
            confidence_threshold: Minimum confidence for detections
        """
        print("=" * 60)
        print("NOVA Object Detection Module - Initializing...")
        print("=" * 60)
        
        self.confidence_threshold = confidence_threshold
        
        # Force CPU-only execution (offline-first requirement)
        self.device = 'cpu'
        print(f"✓ Device: {self.device} (CPU-only, offline-capable)")
        
        # Load pre-trained YOLOv5 model
        # Model will be downloaded to ~/.cache/torch/hub/ultralytics_yolov5_master/
        # on first run, then cached for offline use
        print(f"✓ Loading {model_name} model...")
        start = time.perf_counter()

        #weights_path = r"D:\Major Project\code\tf_models\yolov5n-seg.pt"
        #weights_path = r"yolov5n.pt"
        
        self.model = torch.hub.load('ultralytics/yolov5', model_name, 
                                     pretrained=True, device=self.device)
        self.model.conf = confidence_threshold
        
        load_time = time.perf_counter() - start
        print(f"✓ Model loaded in {load_time:.2f}s")
        
        # Get model file size
        model_path = Path.home() / '.cache/torch/hub/ultralytics_yolov5_master' / f'{model_name}.pt'
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"✓ Model size on disk: {size_mb:.2f} MB")
        
        # COCO class names (80 classes)
        self.class_names = self.model.names
        print(f"✓ Loaded {len(self.class_names)} COCO classes")
        print(f"  Examples: {list(self.class_names.values())[:10]}...")
        print("=" * 60)
    
    def detect_objects(self, image_path=None, frame=None, visualize=True):
        """
        Perform offline object detection on image or video frame.
        
        Args:
            image_path: Path to image file (for static images)
            frame: Numpy array (for video frames)
            visualize: Draw bounding boxes on output
            
        Returns:
            dict: {
                'detections': list of detected objects,
                'inference_time': time in seconds,
                'annotated_image': image with bounding boxes
            }
        """
        # Load image
        if image_path:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
        elif frame is not None:
            img = frame
        else:
            raise ValueError("Provide either image_path or frame")
        
        # Run inference
        start = time.perf_counter()
        results = self.model(img)
        inference_time = time.perf_counter() - start
        
        # Parse results
        detections = []
        df = results.pandas().xyxy[0]  # Bounding boxes as pandas DataFrame
        
        for _, row in df.iterrows():
            detection = {
                'class': row['name'],
                'confidence': float(row['confidence']),
                'bbox': [
                    int(row['xmin']), int(row['ymin']),
                    int(row['xmax']), int(row['ymax'])
                ]
            }
            detections.append(detection)
        
        # Visualize if requested
        annotated_img = img.copy()
        if visualize and len(detections) > 0:
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                label = f"{det['class']}: {det['confidence']:.2f}"
                
                # Draw bounding box
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label background
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(annotated_img, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
                cv2.putText(annotated_img, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return {
            'detections': detections,
            'inference_time': inference_time,
            'annotated_image': annotated_img,
            'num_objects': len(detections)
        }
    
    def process_video(self, video_source=0, max_frames=None):
        """
        Process video stream with real-time FPS calculation.
        
        Args:
            video_source: 0 for webcam, or path to video file
            max_frames: Stop after N frames (None = continuous)
        """
        print("\n" + "=" * 60)
        print("Starting Real-Time Object Detection")
        print("=" * 60)
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {video_source}")
        
        frame_times = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            result = self.detect_objects(frame=frame, visualize=True)
            
            # Calculate FPS
            frame_times.append(result['inference_time'])
            if len(frame_times) > 30:  # Rolling average over 30 frames
                frame_times.pop(0)
            
            avg_time = np.mean(frame_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            # Display info
            info_text = f"FPS: {fps:.1f} | Objects: {result['num_objects']} | Time: {result['inference_time']*1000:.1f}ms"
            cv2.putText(result['annotated_image'], info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show frame
            cv2.imshow('NOVA - Object Detection (Offline)', result['annotated_image'])
            
            # Print periodic stats
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"[Frame {frame_count}] FPS: {fps:.1f} | "
                      f"Avg Inference: {avg_time*1000:.1f}ms | "
                      f"Objects Detected: {result['num_objects']}")
            
            # Check for quit or max frames
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if max_frames and frame_count >= max_frames:
                break
        
        # Cleanup and final stats
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"Total Frames Processed: {frame_count}")
        if frame_times:
            print(f"Average Inference Time: {np.mean(frame_times)*1000:.2f} ms")
            print(f"Average FPS: {1.0/np.mean(frame_times):.2f}")
            print(f"Min Inference Time: {np.min(frame_times)*1000:.2f} ms")
            print(f"Max Inference Time: {np.max(frame_times)*1000:.2f} ms")
        print("=" * 60)


def main():
    """
    Demonstration script for offline object detection.
    """
    # Initialize detector
    detector = OfflineObjectDetector(model_name='yolov5n', confidence_threshold=0.4)
    
    # Example 1: Detect in static image
    print("\nTest 1: Processing static image...")
    test_image = 'test_image.jpg'
    
    if os.path.exists(test_image):
        result = detector.detect_objects(image_path=test_image)
        
        print(f"\nResults:")
        print(f"- Inference Time: {result['inference_time']*1000:.2f} ms")
        print(f"- Objects Found: {result['num_objects']}")
        
        for i, det in enumerate(result['detections'], 1):
            print(f"  {i}. {det['class']}: {det['confidence']:.3f} at {det['bbox']}")
        
        # Save output
        cv2.imwrite('output_detection.jpg', result['annotated_image'])
        print(f"✓ Saved annotated image to: output_detection.jpg")
    else:
        print(f"⚠ Test image '{test_image}' not found. Skipping static test.")
    
    # Example 2: Process webcam (30 frames for testing)
    print("\n\nTest 2: Processing webcam feed (30 frames)...")
    try:
        detector.process_video(video_source=0, max_frames=None)
    except Exception as e:
        print(f"⚠ Webcam test failed: {e}")
        print("  (This is normal if no webcam is connected)")
    
    print("\n✓ Object Detection Module Test Complete")


if __name__ == "__main__":
    """
    EXPECTED OUTPUT (Example):
    
    ============================================================
    NOVA Object Detection Module - Initializing...
    ============================================================
    ✓ Device: cpu (CPU-only, offline-capable)
    ✓ Loading yolov5n model...
    ✓ Model loaded in 2.34s
    ✓ Model size on disk: 3.87 MB
    ✓ Loaded 80 COCO classes
      Examples: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', ...]
    ============================================================
    
    Test 1: Processing static image...
    
    Results:
    - Inference Time: 45.23 ms
    - Objects Found: 3
      1. person: 0.892 at [120, 50, 340, 480]
      2. chair: 0.756 at [450, 200, 620, 450]
      3. bottle: 0.623 at [300, 350, 330, 420]
    ✓ Saved annotated image to: output_detection.jpg
    
    [Frame 30] FPS: 22.3 | Avg Inference: 44.8ms | Objects Detected: 2
    
    ============================================================
    PERFORMANCE SUMMARY
    ============================================================
    Total Frames Processed: 30
    Average Inference Time: 43.67 ms
    Average FPS: 22.91
    Min Inference Time: 39.12 ms
    Max Inference Time: 58.34 ms
    ============================================================
    """
    main()