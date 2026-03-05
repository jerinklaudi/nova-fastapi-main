
import sys
import os
from pathlib import Path

# Add project root to sys.path to ensure 'midas' module can be imported
# Current file: backend/app/models/midas_depth.py
# Root: nova-fastapi-main/
ROOT = Path(__file__).resolve().parents[3]  # backend/app/models/ -> backend/app/ -> backend/ -> nova-fastapi-main/
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np
import torch
import cv2
from typing import List, Optional, Tuple
import time
from app.core.logging import get_logger, log_model_loading, log_inference
from app.core.config import settings
from app.schemas.detection import DepthEstimationResult
from app.utils.image_utils import convert_to_opencv_format, resize_image, normalize_image

logger = get_logger(__name__)

class MiDaSDepthEstimator:
    """MiDaS depth estimation model wrapper - Offline first."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.MIDAS_MODEL_PATH
        self.device = torch.device("cpu")
        self.model = None
        self.transform = None
        self.net_w = None
        self.net_h = None
        self.prev_depth = None
        self.alpha = 0.85  # temporal smoothing factor
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the MiDaS model from local path."""
        try:
            logger.info(f"NOVA Depth Estimation Module - Initializing...")
            logger.info(f"Device: {self.device} (CPU-only, offline-capable)")
            
            # Check if midas module is available
            try:
                from midas.model_loader import load_model
                from midas.transforms import Resize, NormalizeImage, PrepareForNet
                logger.debug("✓ MiDaS module imported successfully")
            except ImportError as ie:
                logger.error(f"✗ MiDaS module not found: {str(ie)}")
                logger.error("  Please ensure the 'midas' package is installed in your Python environment")
                logger.error("  Navigation will continue without depth estimation")
                self.model = None
                return
            
            model_type = "midas_v21_small_256"
            model_path = self.model_path
            
            logger.info(f"Loading MiDaS model ({model_type})...")
            logger.info(f"Model path: {model_path}")
            start = time.perf_counter()
            
            try:
                self.model, self.transform, self.net_w, self.net_h = load_model(
                    device=self.device,
                    model_path=model_path,
                    model_type=model_type,
                    optimize=False
                )
            except FileNotFoundError as fnf:
                logger.error(f"✗ MiDaS model file not found at: {model_path}")
                logger.error("  Please ensure the model file exists at the specified path")
                logger.error("  Navigation will continue without depth estimation")
                self.model = None
                return
            except Exception as load_err:
                logger.error(f"✗ Failed to load MiDaS model: {str(load_err)}")
                logger.error("  Navigation will continue without depth estimation")
                import traceback
                logger.debug(traceback.format_exc())
                self.model = None
                return
            
            load_time = time.perf_counter() - start
            logger.info(f"✓ Model loaded successfully in {load_time:.2f}s")
            logger.info(f"  Model outputs RELATIVE depth")
            logger.info(f"  Input size: {self.net_w}x{self.net_h}")
            
            log_model_loading("MiDaS Depth", True)
            
        except Exception as e:
            log_model_loading("MiDaS Depth", False, str(e))
            logger.error(f"✗ Failed to load MiDaS model: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            self.model = None
    
    def enhance_lighting(self, img_bgr: np.ndarray) -> np.ndarray:
        """Enhance lighting using CLAHE for better depth estimation."""
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
    
    def estimate_depth(self, image: np.ndarray) -> DepthEstimationResult:
        """Estimate depth from input image."""
        start_time = time.perf_counter()
        
        try:
            if self.model is None:
                logger.warning("MiDaS model not loaded - returning None for degraded mode")
                # Return None instead of raising exception to allow degraded mode
                return None
            
            # Enhance lighting
            image = self.enhance_lighting(image)
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Preprocess image
            input_tensor = self.transform({"image": img_rgb})["image"]
            input_batch = torch.from_numpy(input_tensor).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                prediction = self.model(input_batch)
                
                # Interpolate to original resolution
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image.shape[:2],
                    mode='bicubic',
                    align_corners=False
                ).squeeze().cpu().numpy()
            
            inference_time = time.perf_counter() - start_time
            
            # Convert to numpy if still a tensor
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

            # Normalize to [0, 1] range with perceptual boost
            depth_clipped = np.clip(depth_map, near, far)
            depth_normalized = (depth_clipped - near) / (far - near + 1e-6)
            depth_normalized = np.sqrt(depth_normalized)  # perceptual boost
            depth_normalized = 1.0 - depth_normalized
            
            # Log inference details
            log_inference("MiDaS Depth", inference_time, image.shape, depth_map.shape)
            
            # Create result
            result = DepthEstimationResult(
                min_depth=float(near),
                max_depth=float(far),
                mean_depth=float(depth_map.mean()),
                inference_time_ms=round(inference_time * 1000, 2)
            )
            
            logger.info(f"Depth estimation completed: {result.inference_time_ms}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Depth estimation failed: {str(e)}")
            raise
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_path": self.model_path,
            "model_type": "MiDaS v2.1 Small",
            "input_size": f"{self.net_w}x{self.net_h}" if self.net_w else "N/A",
            "device": str(self.device),
            "temporal_smoothing": True,
            "lighting_enhancement": True
        }

