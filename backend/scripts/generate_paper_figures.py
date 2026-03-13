import argparse
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests


plt.rcParams.update(
    {
        "figure.dpi": 170,
        "savefig.dpi": 320,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    }
)


def read_image_rgb(image_path: Path) -> np.ndarray:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def post_image(
    endpoint: str,
    image_path: Path,
    params: Dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> Tuple[Dict[str, Any], float]:
    suffix = image_path.suffix.lower()
    if suffix == ".png":
        mime = "image/png"
    else:
        mime = "image/jpeg"

    with image_path.open("rb") as f:
        files = {
            "file": (
                image_path.name,
                f.read(),
                mime,
            )
        }

    start = time.perf_counter()
    resp = requests.post(endpoint, files=files, params=params or {}, timeout=timeout)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    if resp.status_code != 200:
        raise RuntimeError(f"{endpoint} failed ({resp.status_code}): {resp.text[:500]}")

    return resp.json(), elapsed_ms


def draw_detection_boxes(image_rgb: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    out = image_rgb.copy()
    h, w = out.shape[:2]

    for det in detections:
        bbox = det.get("bbox", {})
        x1 = int(float(bbox.get("left", 0.0)) * w)
        y1 = int(float(bbox.get("top", 0.0)) * h)
        x2 = int(float(bbox.get("right", 0.0)) * w)
        y2 = int(float(bbox.get("bottom", 0.0)) * h)

        label = str(det.get("label", "obj"))
        conf = float(det.get("confidence", 0.0))
        text = f"{label} {conf:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), (20, 255, 60), 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), (20, 255, 60), -1)
        cv2.putText(out, text, (x1 + 4, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (15, 15, 15), 2)

    return out


def draw_ocr_boxes(image_rgb: np.ndarray, text_regions: List[Dict[str, Any]]) -> np.ndarray:
    out = image_rgb.copy()
    h, w = out.shape[:2]

    for region in text_regions:
        bbox = region.get("bbox", {})
        x1 = int(float(bbox.get("left", 0.0)) * w)
        y1 = int(float(bbox.get("top", 0.0)) * h)
        x2 = int(float(bbox.get("right", 0.0)) * w)
        y2 = int(float(bbox.get("bottom", 0.0)) * h)

        txt = str(region.get("text", "")).strip()
        conf = float(region.get("confidence", 0.0))
        label = f"{txt[:28]} ({conf:.2f})" if txt else f"text ({conf:.2f})"

        cv2.rectangle(out, (x1, y1), (x2, y2), (40, 140, 255), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), (40, 140, 255), -1)
        cv2.putText(out, label, (x1 + 3, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return out


def depth_to_colormap(depth_map: List[List[float]]) -> np.ndarray:
    depth = np.array(depth_map, dtype=np.float32)
    if depth.size == 0:
        raise ValueError("Depth map is empty")

    lo, hi = np.percentile(depth, [2, 98])
    depth = np.clip(depth, lo, hi)
    depth = (depth - lo) / (hi - lo + 1e-9)
    depth_u8 = (depth * 255).astype(np.uint8)
    depth_bgr = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB)


def depth_local_fallback(repo_root: Path, image_path: Path) -> Tuple[Dict[str, Any], float]:
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    backend_dir = repo_root / "backend"
    if str(backend_dir) not in sys.path:
        sys.path.append(str(backend_dir))

    from backend.app.models.midas_depth import MiDaSDepthEstimator

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image for local depth fallback: {image_path}")

    estimator = MiDaSDepthEstimator()
    start = time.perf_counter()
    result = estimator.estimate_depth(image_bgr)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    if result is None:
        raise RuntimeError("Local MiDaS fallback failed: estimator returned None")

    raw_depth = estimator.prev_depth
    if raw_depth is None:
        raise RuntimeError("Local MiDaS fallback failed: no depth map available from estimator")

    near, far = np.percentile(raw_depth, (5, 95))
    depth_clipped = np.clip(raw_depth, near, far)
    depth_norm = (depth_clipped - near) / (far - near + 1e-6)
    depth_norm = np.sqrt(depth_norm)
    depth_norm = 1.0 - depth_norm

    return {
        "depth_map": depth_norm.tolist(),
        "min_depth": result.min_depth,
        "max_depth": result.max_depth,
        "mean_depth": result.mean_depth,
        "inference_time_ms": result.inference_time_ms,
    }, elapsed_ms


def post_depth_with_fallback(depth_url: str, repo_root: Path, image_path: Path) -> Tuple[Dict[str, Any], float]:
    try:
        return post_image(depth_url, image_path)
    except Exception as exc:
        print(f"[WARN] /detect/depth failed, using local MiDaS fallback: {exc}")
        return depth_local_fallback(repo_root, image_path)


def ocr_local_fallback(repo_root: Path, image_path: Path) -> Tuple[Dict[str, Any], float]:
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    backend_dir = repo_root / "backend"
    if str(backend_dir) not in sys.path:
        sys.path.append(str(backend_dir))

    from backend.app.models.paddle_ocr import PaddleOCRDetector

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image for local OCR fallback: {image_path}")

    detector = PaddleOCRDetector()
    start = time.perf_counter()
    blocks = detector.recognize_text(image_bgr)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    regions = []
    for b in blocks:
        regions.append(
            {
                "text": b.text,
                "confidence": b.confidence,
                "bbox": {
                    "left": b.bbox.left,
                    "top": b.bbox.top,
                    "right": b.bbox.right,
                    "bottom": b.bbox.bottom,
                },
            }
        )

    return {
        "text_regions": regions,
        "inference_time_ms": elapsed_ms,
    }, elapsed_ms


def post_ocr_with_fallback(ocr_url: str, repo_root: Path, image_path: Path) -> Tuple[Dict[str, Any], float]:
    try:
        return post_image(ocr_url, image_path)
    except Exception as exc:
        print(f"[WARN] /detect/text failed, using local OCR fallback: {exc}")
        return ocr_local_fallback(repo_root, image_path)


def render_figure_2(output_dir: Path, parallel_ms: float, ocr_ms: float) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 2.9))
    ax.set_title("Figure 2: Temporal Pipeline")
    ax.set_xlabel("Time (ms)")
    ax.set_yticks([0])
    ax.set_yticklabels(["Frame t"])

    ax.broken_barh([(0, parallel_ms)], (-0.25, 0.5), facecolors="#2E86AB", label=f"Detection + Depth (parallel): {parallel_ms:.1f} ms")
    ax.broken_barh([(parallel_ms + 20, ocr_ms)], (-0.25, 0.5), facecolors="#F18F01", label=f"OCR on-demand: {ocr_ms:.1f} ms")

    ax.axvline(parallel_ms, color="#4a4a4a", linestyle="--", linewidth=1)
    ax.text(parallel_ms + 5, 0.33, f"~{parallel_ms:.0f} ms", fontsize=10)
    ax.text(parallel_ms + 25, -0.37, f"~{ocr_ms:.0f} ms", fontsize=10)

    xmax = parallel_ms + ocr_ms + 80
    ax.set_xlim(0, xmax)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    fig.savefig(output_dir / "figure2_temporal_pipeline.png", bbox_inches="tight")
    plt.close(fig)


def render_figure_3(
    output_dir: Path,
    input_rgb: np.ndarray,
    det_rgb: np.ndarray,
    depth_rgb: np.ndarray,
    ocr_rgb: np.ndarray,
    ocr_lines: List[str],
) -> None:
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.16, wspace=0.07)
    fig.suptitle("Figure 3: Multi-Modal Perception on a Real Urban Scene", fontsize=15, y=0.98)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(input_rgb)
    ax1.set_title("(a) Input image")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(det_rgb)
    ax2.set_title("(b) YOLO object detections")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(depth_rgb)
    ax3.set_title("(c) MiDaS depth estimation")
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(ocr_rgb)
    ax4.set_title("(d) OCR output")
    ax4.axis("off")

    if ocr_lines:
        shown = "\n".join(ocr_lines[:4])
        ax4.text(
            0.02,
            0.98,
            shown,
            transform=ax4.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color="white",
            bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", boxstyle="round,pad=0.35"),
        )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    fig.savefig(output_dir / "figure3_multimodal_perception.png", bbox_inches="tight")
    plt.close(fig)


def render_figure_4(
    output_dir: Path,
    face_count: int,
    identities: List[str],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle("Figure 4: System Flows", fontsize=15, y=0.98)

    ax = axes[0]
    ax.set_title("(a) Face recognition cascade")
    ax.axis("off")

    x = [0.05, 0.30, 0.55, 0.80]
    labels = [
        "YOLOv8n-Face\nDetection",
        "Landmark\nAlignment",
        "SFace\nEmbedding",
        "Database\nMatching",
    ]
    colors = ["#4C78A8", "#72B7B2", "#54A24B", "#E45756"]
    for i, (xi, lbl, col) in enumerate(zip(x, labels, colors)):
        ax.add_patch(plt.Rectangle((xi, 0.43), 0.16, 0.22, color=col, alpha=0.92, ec="black", lw=1.0))
        ax.text(xi + 0.08, 0.54, lbl, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        if i < len(x) - 1:
            ax.annotate("", xy=(x[i + 1], 0.54), xytext=(xi + 0.16, 0.54), arrowprops=dict(arrowstyle="->", lw=2))

    id_txt = ", ".join(identities[:4]) if identities else "Unknown"
    ax.text(0.05, 0.18, f"Detections on image.png: {face_count}", fontsize=10)
    ax.text(0.05, 0.10, f"Recognized identities: {id_txt}", fontsize=10)

    ax2 = axes[1]
    ax2.set_title("(b) Emergency SOS system flow")
    ax2.axis("off")

    flow_nodes = [
        (0.08, 0.72, 0.26, 0.16, "Hardware/Voice\nTrigger"),
        (0.40, 0.72, 0.26, 0.16, "Suspend ML\nInference"),
        (0.72, 0.72, 0.24, 0.16, "Capture GPS"),
        (0.22, 0.40, 0.30, 0.16, "Send SMS to\nEmergency Contacts"),
        (0.62, 0.40, 0.30, 0.16, "Place Emergency\nPhone Call"),
        (0.40, 0.10, 0.26, 0.16, "Restart\nNavigation"),
    ]

    for x0, y0, w, h, text in flow_nodes:
        ax2.add_patch(plt.Rectangle((x0, y0), w, h, color="#3B3B3B", ec="#101010", lw=1.1, alpha=0.95))
        ax2.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=10, color="white")

    arrows = [
        ((0.34, 0.80), (0.40, 0.80)),
        ((0.66, 0.80), (0.72, 0.80)),
        ((0.84, 0.72), (0.77, 0.56)),
        ((0.50, 0.72), (0.37, 0.56)),
        ((0.37, 0.40), (0.46, 0.26)),
        ((0.77, 0.40), (0.58, 0.26)),
    ]
    for start, end in arrows:
        ax2.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=2, color="#2b2b2b"))

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_dir / "figure4_flows.png", bbox_inches="tight")
    plt.close(fig)


def render_figure_5(
    output_dir: Path,
    object_latencies: List[float],
    depth_latencies: List[float],
    object_conf: List[float],
    depth_quality: List[float],
    ocr_conf: List[float],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    fig.suptitle("Figure 5: Latency Stability and Confidence Distributions", fontsize=15, y=0.98)

    ax = axes[0]
    frames = np.arange(1, len(object_latencies) + 1)
    ax.plot(frames, object_latencies, marker="o", markersize=3.6, linewidth=1.4, label="Object detection")
    ax.plot(frames, depth_latencies, marker="s", markersize=3.6, linewidth=1.4, label="Depth estimation")
    ax.set_title("(a) 30-frame latency stability")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Latency (ms)")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend()

    ax2 = axes[1]
    bins = np.linspace(0.0, 1.0, 21)
    if object_conf:
        ax2.hist(object_conf, bins=bins, alpha=0.65, label="Object confidence", color="#2E86AB")
    if depth_quality:
        ax2.hist(depth_quality, bins=bins, alpha=0.65, label="Depth quality", color="#54A24B")
    if ocr_conf:
        ax2.hist(ocr_conf, bins=bins, alpha=0.65, label="OCR confidence", color="#F18F01")
    ax2.set_title("(b) Confidence distributions")
    ax2.set_xlabel("Score")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, linestyle=":", alpha=0.35)
    ax2.legend()

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_dir / "figure5_latency_confidence.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate research figures using NOVA backend models")
    parser.add_argument("--backend-url", default="http://10.62.137.189:8000", help="FastAPI backend base URL")
    parser.add_argument(
        "--image",
        default="models/image.png",
        help="Input image for inference",
    )
    parser.add_argument(
        "--output-dir",
        default="paper_figures_output",
        help="Folder to store generated figures",
    )
    parser.add_argument("--frames", type=int, default=30, help="Frames for Figure 5 latency stability")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    image_path = (repo_root / args.image).resolve()
    out_dir = (repo_root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    detect_url = f"{args.backend_url}/detect/objects"
    depth_url = f"{args.backend_url}/detect/depth"
    ocr_url = f"{args.backend_url}/detect/text"
    face_url = f"{args.backend_url}/detect/faces"

    input_rgb = read_image_rgb(image_path)

    # Warmup once for fair timing
    post_image(detect_url, image_path)
    post_depth_with_fallback(depth_url, repo_root, image_path)
    post_ocr_with_fallback(ocr_url, repo_root, image_path)

    # Figure 2 timings
    parallel_runs = []
    ocr_runs = []
    for _ in range(5):
        with ThreadPoolExecutor(max_workers=2) as ex:
            obj_future = ex.submit(post_image, detect_url, image_path)
            dep_future = ex.submit(post_depth_with_fallback, depth_url, repo_root, image_path)
            _, obj_ms = obj_future.result()
            _, dep_ms = dep_future.result()
        parallel_runs.append(max(obj_ms, dep_ms))

        _, ocr_ms = post_ocr_with_fallback(ocr_url, repo_root, image_path)
        ocr_runs.append(ocr_ms)

    parallel_mean = float(np.mean(parallel_runs))
    ocr_mean = float(np.mean(ocr_runs))
    render_figure_2(out_dir, parallel_mean, ocr_mean)

    # Figure 3 data
    object_json, _ = post_image(detect_url, image_path)
    depth_json, _ = post_depth_with_fallback(depth_url, repo_root, image_path)
    ocr_json, _ = post_ocr_with_fallback(ocr_url, repo_root, image_path)

    detections = object_json.get("detections", [])
    text_regions = ocr_json.get("text_regions", [])
    ocr_lines = [str(r.get("text", "")).strip() for r in text_regions if str(r.get("text", "")).strip()]

    det_rgb = draw_detection_boxes(input_rgb, detections)
    depth_rgb = depth_to_colormap(depth_json.get("depth_map", []))
    ocr_rgb = draw_ocr_boxes(input_rgb, text_regions)

    cv2.imwrite(str(out_dir / "figure3b_object_detections.png"), cv2.cvtColor(det_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "figure3c_depth_map.png"), cv2.cvtColor(depth_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "figure3d_ocr_output.png"), cv2.cvtColor(ocr_rgb, cv2.COLOR_RGB2BGR))
    render_figure_3(out_dir, input_rgb, det_rgb, depth_rgb, ocr_rgb, ocr_lines)

    # Figure 4 data from real face endpoint
    faces_json, _ = post_image(face_url, image_path, params={"recognize_faces": "true"})
    faces = faces_json.get("faces", [])
    identities = []
    for f in faces:
        pid = f.get("person_id")
        if isinstance(pid, str) and pid and pid.lower() != "unknown":
            identities.append(pid)
    render_figure_4(out_dir, face_count=len(faces), identities=sorted(set(identities)))

    # Figure 5 data
    object_latencies: List[float] = []
    depth_latencies: List[float] = []
    object_conf: List[float] = []
    depth_quality: List[float] = []
    ocr_conf: List[float] = []

    for _ in range(args.frames):
        obj_json, obj_ms = post_image(detect_url, image_path)
        dep_json, dep_ms = post_depth_with_fallback(depth_url, repo_root, image_path)
        txt_json, _ = post_ocr_with_fallback(ocr_url, repo_root, image_path)

        object_latencies.append(obj_ms)
        depth_latencies.append(dep_ms)

        for det in obj_json.get("detections", []):
            c = det.get("confidence", None)
            if isinstance(c, (int, float)):
                object_conf.append(float(c))

        dmap = np.array(dep_json.get("depth_map", []), dtype=np.float32)
        if dmap.size > 0:
            spread = float(np.std(dmap))
            spread_clipped = max(0.0, min(1.0, spread))
            depth_quality.append(spread_clipped)

        for region in txt_json.get("text_regions", []):
            c = region.get("confidence", None)
            if isinstance(c, (int, float)):
                ocr_conf.append(float(c))

    render_figure_5(out_dir, object_latencies, depth_latencies, object_conf, depth_quality, ocr_conf)

    summary = {
        "image": str(image_path),
        "backend_url": args.backend_url,
        "frames": args.frames,
        "parallel_mean_ms": round(parallel_mean, 2),
        "ocr_mean_ms": round(ocr_mean, 2),
        "object_latency_ms": {
            "mean": round(float(np.mean(object_latencies)), 2),
            "std": round(float(np.std(object_latencies)), 2),
            "min": round(float(np.min(object_latencies)), 2),
            "max": round(float(np.max(object_latencies)), 2),
        },
        "depth_latency_ms": {
            "mean": round(float(np.mean(depth_latencies)), 2),
            "std": round(float(np.std(depth_latencies)), 2),
            "min": round(float(np.min(depth_latencies)), 2),
            "max": round(float(np.max(depth_latencies)), 2),
        },
        "num_object_conf_samples": len(object_conf),
        "num_depth_quality_samples": len(depth_quality),
        "num_ocr_conf_samples": len(ocr_conf),
    }

    with (out_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Generated paper figures in:", out_dir)
    print("- figure2_temporal_pipeline.png")
    print("- figure3_multimodal_perception.png")
    print("- figure3b_object_detections.png")
    print("- figure3c_depth_map.png")
    print("- figure3d_ocr_output.png")
    print("- figure4_flows.png")
    print("- figure5_latency_confidence.png")
    print("- metrics_summary.json")


if __name__ == "__main__":
    main()
