"""
NOVA - Depth Estimation Module (Offline-First)
===============================================
B.Tech Research Project - Assistive Vision System

WHY THIS MODEL:
- MiDaS Small v2.1: Lightweight monocular depth estimation
- Size: ~100 MB (optimized for mobile/CPU deployment)
- Provides relative depth (sufficient for obstacle detection)
- No need for absolute distance measurements
- Fully offline after initial model download

PERFORMANCE TARGET:
- CPU-only inference: 5-10 FPS on modern laptop
- Inference time: ~100-200 ms per frame
- Model size: 100 MB (well under 400 MB limit)

ACADEMIC JUSTIFICATION:
- MiDaS is state-of-the-art for monocular depth estimation
- Widely used in assistive technology research
- Provides relative depth sufficient for navigation assistance
- Pre-trained on diverse datasets (robust generalization)
"""

import torch
import cv2
import numpy as np
import time
from pathlib import Path

from midas.model_loader import load_model
from midas.transforms import Resize, NormalizeImage, PrepareForNet




class OfflineDepthEstimator:
    """
    Offline-first depth estimation using MiDaS Small.
    Provides relative depth maps for obstacle detection.
    """
    
    def __init__(self):
        print("=" * 60)
        print("NOVA Depth Estimation Module - Initializing...")
        print("=" * 60)

        self.device = torch.device("cpu")
        print(f"✓ Device: {self.device} (CPU-only, offline-capable)")

        model_type = "midas_v21_small_256"
        model_path = "midas_v21_small_256.pt"  # MUST match repo structure

        print(f"✓ Loading MiDaS model ({model_type})...")
        start = time.perf_counter()

        self.model, self.transform, self.net_w, self.net_h = load_model(
            device=self.device,
            model_path=model_path,
            model_type=model_type,
            optimize=False
        )
        self.prev_depth = None
        self.alpha = 0.85  # temporal smoothing factor


        load_time = time.perf_counter() - start
        print(f"✓ Model loaded in {load_time:.2f}s")

        size_mb = Path(model_path).stat().st_size / (1024 * 1024)
        print(f"✓ Model size on disk: {size_mb:.2f} MB")

        print("✓ Model outputs RELATIVE depth")
        print("=" * 60)

    def enhance_lighting(self, img_bgr):
        # Convert to LAB color space
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # CLAHE on L-channel (adaptive contrast)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Merge back
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return enhanced


    def estimate_depth(self, image_path=None, frame=None, visualize=True):
        """
        Perform offline depth estimation.
        
        Args:
            image_path: Path to image file
            frame: Numpy array (BGR format from OpenCV)
            visualize: Generate depth heatmap
            
        Returns:
            dict: {
                'depth_map': relative depth as numpy array,
                'inference_time': time in seconds,
                'depth_visualization': colored depth map (if visualize=True),
                'min_depth': minimum depth value,
                'max_depth': maximum depth value
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
        
        # Convert BGR to RGB
        
        img = self.enhance_lighting(img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Preprocess image
        input_tensor = self.transform({"image": img_rgb})["image"]
        input_batch = torch.from_numpy(input_tensor).unsqueeze(0).to(self.device)
        
        # Run inference
        start = time.perf_counter()
        with torch.no_grad():
            prediction = self.model(input_batch)
            
            # Interpolate to original resolution (Moved inside or right after no_grad)
            if isinstance(prediction, torch.Tensor):
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode='bicubic',
                    align_corners=False
                ).squeeze().cpu().numpy()
        
        inference_time = time.perf_counter() - start
        
        # Convert to numpy (Ensuring prediction is handled if it's still a tensor)
        if isinstance(prediction, torch.Tensor):
            depth_map = prediction.cpu().numpy()
        else:
            depth_map = prediction

        # Temporal smoothing to reduce flicker and lighting noise
        if self.prev_depth is not None:
            depth_map = self.alpha * self.prev_depth + (1 - self.alpha) * depth_map

        self.prev_depth = depth_map

        
        # Normalize for visualization (robust + high contrast)
        # Robust depth range using percentiles (avoids outliers)
        if np.all(depth_map == depth_map.flat[0]):
            depth_map = depth_map + 1e-6
        else:
            depth_map = depth_map

        near, far = np.percentile(depth_map, (5, 95))

        result = {
            'depth_map': depth_map,
            'inference_time': inference_time,
            'min_depth': float(near),
            'max_depth': float(far),
            'depth_visualization': None
        }

        # Create visualization if requested
        if visualize:
            # Clip extreme values
            depth_clipped = np.clip(depth_map, near, far)
            depth_normalized = (depth_clipped - near) / (far - near + 1e-6)
            depth_normalized = np.sqrt(depth_normalized)  # 👈 perceptual boost
            depth_normalized = 1.0 - depth_normalized
            depth_8bit = (depth_normalized * 255).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_TURBO)

        return result

    
    def create_depth_comparison(self, image_path=None, frame=None, save_path='depth_comparison.jpg'):
        """
        Create side-by-side comparison of original image and depth map.
        
        Args:
            image_path: Path to input image
            frame: Or provide frame directly
            save_path: Where to save the comparison
        """
        # Load original image
        if image_path:
            original = cv2.imread(image_path)
        elif frame is not None:
            original = frame
        else:
            raise ValueError("Provide either image_path or frame")
        
        # Estimate depth
        result = self.estimate_depth(image_path=image_path, frame=frame, visualize=True)
        
        # Resize depth map to match original
        depth_vis = cv2.resize(result['depth_visualization'], 
                               (original.shape[1], original.shape[0]))
        
        # Create side-by-side comparison
        comparison = np.hstack([original, depth_vis])
        
        # Add labels
        cv2.putText(comparison, "Original Image", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Depth Map (Dark=Close)", (original.shape[1] + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add performance info
        info = f"Inference: {result['inference_time']*1000:.1f}ms"
        cv2.putText(comparison, info, (10, comparison.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Save
        cv2.imwrite(save_path, comparison)
        print(f"✓ Saved depth comparison to: {save_path}")
        
        return comparison, result
    
    def process_video(self, video_source=0, max_frames=None):
        """
        Process video stream with real-time depth estimation.
        
        Args:
            video_source: 0 for webcam, or path to video file
            max_frames: Stop after N frames (None = continuous)
        """
        print("\n" + "=" * 60)
        print("Starting Real-Time Depth Estimation")
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
            
            # Estimate depth
            result = self.estimate_depth(frame=frame, visualize=True)
            
            # Calculate FPS
            frame_times.append(result['inference_time'])
            if len(frame_times) > 30:
                frame_times.pop(0)
            
            avg_time = np.mean(frame_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            # Create side-by-side display
            frame_resized = cv2.resize(frame, (result['depth_visualization'].shape[1], 
                                               result['depth_visualization'].shape[0]))
            display = np.hstack([frame_resized, result['depth_visualization']])
            
            # Display info
            info_text = f"FPS: {fps:.1f} | Time: {result['inference_time']*1000:.1f}ms"
            cv2.putText(display, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, "Original", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, "Depth (Dark=Near)", (frame_resized.shape[1] + 10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('NOVA - Depth Estimation (Offline)', display)
            
            # Print periodic stats
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"[Frame {frame_count}] FPS: {fps:.1f} | "
                      f"Avg Inference: {avg_time*1000:.1f}ms")
            
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
    Demonstration script for offline depth estimation.
    """
    # Initialize estimator
    estimator = OfflineDepthEstimator()
    
    # Example 1: Process static image
    print("\nTest 1: Processing static image...")
    test_image = 'test_image.jpg'
    
    if os.path.exists(test_image):
        comparison, result = estimator.create_depth_comparison(
            image_path=test_image, 
            save_path='output_depth.jpg'
        )
        
        print(f"\nResults:")
        print(f"- Inference Time: {result['inference_time']*1000:.2f} ms")
        print(f"- Depth Range: {result['min_depth']:.2f} to {result['max_depth']:.2f}")
        print(f"- Resolution: {result['depth_map'].shape}")
    else:
        print(f"⚠ Test image '{test_image}' not found. Skipping static test.")
    
    # Example 2: Process webcam (30 frames for testing)
    print("\n\nTest 2: Processing webcam feed (30 frames)...")
    try:
        estimator.process_video(video_source=0)
    except Exception as e:
        print(f"⚠ Webcam test failed: {e}")
        print("  (This is normal if no webcam is connected)")
    
    print("\n✓ Depth Estimation Module Test Complete")


if __name__ == "__main__":
    """
    EXPECTED OUTPUT (Example):
    
    ============================================================
    NOVA Depth Estimation Module - Initializing...
    ============================================================
    ✓ Device: cpu (CPU-only, offline-capable)
    ✓ Loading MiDaS_small model...
    ✓ Model loaded in 3.12s
    ✓ Model size on disk: 102.45 MB
    ✓ Model outputs RELATIVE depth (closer = lower values)
      No calibration needed for obstacle detection
    ============================================================
    
    Test 1: Processing static image...
    ✓ Saved depth comparison to: output_depth.jpg
    
    Results:
    - Inference Time: 145.67 ms
    - Depth Range: 0.23 to 125.89
    - Resolution: (480, 640)
    
    [Frame 30] FPS: 7.2 | Avg Inference: 138.9ms
    
    ============================================================
    PERFORMANCE SUMMARY
    ============================================================
    Total Frames Processed: 30
    Average Inference Time: 142.34 ms
    Average FPS: 7.03
    Min Inference Time: 128.45 ms
    Max Inference Time: 167.23 ms
    ============================================================
    """
    import os
    main()