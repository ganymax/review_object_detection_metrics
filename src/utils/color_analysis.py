"""
Color analysis utilities for bounding box bias profiling.

Provides dominant color extraction and center of gravity calculation
for medium and large bounding boxes (small boxes are excluded).
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from src.utils.object_scale import ObjectScale, classify_scale


@dataclass
class ColorAnalysisResult:
    """Result of color analysis for a bounding box region."""
    dominant_color_rgb: Tuple[int, int, int]
    center_of_gravity: Tuple[int, int]  # (x, y) in image coordinates
    mask_pixel_count: int
    roi_bounds: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    scale: ObjectScale
    
    @property
    def dominant_color_bgr(self) -> Tuple[int, int, int]:
        """Get dominant color in BGR format for OpenCV."""
        return (self.dominant_color_rgb[2], self.dominant_color_rgb[1], self.dominant_color_rgb[0])
    
    @property
    def contrasting_color_rgb(self) -> Tuple[int, int, int]:
        """Get black or white based on dominant color luminance."""
        r, g, b = self.dominant_color_rgb
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return (0, 0, 0) if luminance > 128 else (255, 255, 255)
    
    @property
    def contrasting_color_bgr(self) -> Tuple[int, int, int]:
        """Get contrasting color in BGR format."""
        rgb = self.contrasting_color_rgb
        return (rgb[2], rgb[1], rgb[0])


def extract_roi(image: np.ndarray, bbox_coords: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
    """
    Extract region of interest from image given bounding box coordinates.
    
    Args:
        image: Input image (RGB or BGR, HxWxC).
        bbox_coords: Bounding box as (x1, y1, x2, y2).
    
    Returns:
        Cropped ROI or None if invalid.
    """
    if image is None or len(image.shape) < 2:
        return None
    
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox_coords
    
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    return image[y1:y2, x1:x2].copy()


def find_dominant_color(roi: np.ndarray, color_tolerance: int = 30, min_cluster_size: int = 8) -> Tuple[int, int, int]:
    """
    Find the single most dominant RGB color in a region.
    
    Uses k-means clustering with k=5 and returns the color of the largest cluster.
    
    Args:
        roi: Region of interest image (RGB format).
        color_tolerance: Not used in k-means approach but kept for API compatibility.
        min_cluster_size: Minimum number of clusters for k-means.
    
    Returns:
        Dominant color as (R, G, B) tuple.
    """
    if roi is None or roi.size == 0:
        return (128, 128, 128)
    
    if len(roi.shape) == 2:
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
    
    pixels = roi.reshape(-1, 3).astype(np.float32)
    
    if len(pixels) < min_cluster_size:
        mean_color = np.mean(pixels, axis=0)
        return (int(mean_color[0]), int(mean_color[1]), int(mean_color[2]))
    
    n_clusters = min(5, len(pixels) // min_cluster_size)
    n_clusters = max(1, n_clusters)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    
    try:
        _, labels, centers = cv2.kmeans(
            pixels, n_clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
    except cv2.error:
        mean_color = np.mean(pixels, axis=0)
        return (int(mean_color[0]), int(mean_color[1]), int(mean_color[2]))
    
    unique, counts = np.unique(labels, return_counts=True)
    dominant_idx = unique[np.argmax(counts)]
    dominant_center = centers[dominant_idx]
    
    return (
        int(np.clip(dominant_center[0], 0, 255)),
        int(np.clip(dominant_center[1], 0, 255)),
        int(np.clip(dominant_center[2], 0, 255))
    )


def create_color_mask(roi: np.ndarray, target_color: Tuple[int, int, int], tolerance: int = 40) -> np.ndarray:
    """
    Create a binary mask for pixels matching the target color within tolerance.
    
    Args:
        roi: Region of interest (RGB format).
        target_color: Target color as (R, G, B).
        tolerance: Color distance tolerance.
    
    Returns:
        Binary mask (255 for matching pixels, 0 otherwise).
    """
    if roi is None or roi.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    
    if len(roi.shape) == 2:
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
    
    target = np.array(target_color, dtype=np.float32)
    pixels = roi.astype(np.float32)
    
    diff = np.sqrt(np.sum((pixels - target) ** 2, axis=2))
    mask = (diff <= tolerance).astype(np.uint8) * 255
    
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask


def calculate_center_of_gravity(mask: np.ndarray) -> Tuple[int, int]:
    """
    Calculate center of gravity (centroid) of a binary mask.
    
    Args:
        mask: Binary mask image.
    
    Returns:
        (x, y) coordinates of the center of gravity relative to mask origin.
        Returns mask center if no valid pixels found.
    """
    if mask is None or mask.size == 0:
        return (0, 0)
    
    h, w = mask.shape[:2]
    
    moments = cv2.moments(mask)
    
    if moments['m00'] > 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))
        return (cx, cy)
    
    return (w // 2, h // 2)


def analyze_bounding_box_color(
    image: np.ndarray,
    bbox_coords: Tuple[float, float, float, float],
    color_tolerance: int = 40
) -> Optional[ColorAnalysisResult]:
    """
    Analyze a bounding box to find dominant color and its center of gravity.
    
    Only processes medium and large bounding boxes. Small boxes are skipped.
    
    Args:
        image: Input image (RGB format, HxWxC).
        bbox_coords: Bounding box as (x1, y1, x2, y2) in absolute coordinates.
        color_tolerance: Tolerance for color matching when creating mask.
    
    Returns:
        ColorAnalysisResult or None if box is too small or invalid.
    """
    if image is None:
        return None
    
    x1, y1, x2, y2 = bbox_coords
    area = (x2 - x1) * (y2 - y1)
    scale = classify_scale(area)
    
    if scale == ObjectScale.SMALL or scale == ObjectScale.UNKNOWN:
        return None
    
    roi = extract_roi(image, bbox_coords)
    if roi is None:
        return None
    
    dominant_color = find_dominant_color(roi)
    mask = create_color_mask(roi, dominant_color, tolerance=color_tolerance)
    local_cog = calculate_center_of_gravity(mask)
    
    x1_int, y1_int = int(x1), int(y1)
    global_cog = (x1_int + local_cog[0], y1_int + local_cog[1])
    
    mask_count = int(np.sum(mask > 0))
    
    return ColorAnalysisResult(
        dominant_color_rgb=dominant_color,
        center_of_gravity=global_cog,
        mask_pixel_count=mask_count,
        roi_bounds=(int(x1), int(y1), int(x2), int(y2)),
        scale=scale
    )


def analyze_bounding_box_from_bb(
    image: np.ndarray,
    bb,
    color_tolerance: int = 40
) -> Optional[ColorAnalysisResult]:
    """
    Analyze a BoundingBox object for dominant color and center of gravity.
    
    Args:
        image: Input image (RGB format).
        bb: BoundingBox object.
        color_tolerance: Tolerance for color matching.
    
    Returns:
        ColorAnalysisResult or None if box is too small or invalid.
    """
    try:
        from src.utils.enumerators import BBFormat
        coords = bb.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
        return analyze_bounding_box_color(image, coords, color_tolerance)
    except Exception:
        return None


def draw_crosshair_marker(
    image: np.ndarray,
    center: Tuple[int, int],
    fill_color_bgr: Tuple[int, int, int],
    border_color_bgr: Tuple[int, int, int],
    size: int = 10,
    thickness: int = 2
) -> np.ndarray:
    """
    Draw a crosshair marker at the specified position.
    
    The marker consists of:
    - A center circle filled with the dominant color
    - Cross lines extending from the circle
    - Black/white border for contrast
    
    Args:
        image: Image to draw on (modified in place and returned).
        center: (x, y) center coordinates.
        fill_color_bgr: Fill color in BGR format.
        border_color_bgr: Border color in BGR format.
        size: Marker size (radius of central circle).
        thickness: Line thickness for cross arms.
    
    Returns:
        Image with marker drawn.
    """
    if image is None:
        return image
    
    h, w = image.shape[:2]
    cx, cy = center
    
    cx = max(size + thickness, min(w - size - thickness - 1, cx))
    cy = max(size + thickness, min(h - size - thickness - 1, cy))
    
    arm_length = size + 5
    
    cv2.line(image, (cx - arm_length, cy), (cx - size, cy), border_color_bgr, thickness + 2)
    cv2.line(image, (cx + size, cy), (cx + arm_length, cy), border_color_bgr, thickness + 2)
    cv2.line(image, (cx, cy - arm_length), (cx, cy - size), border_color_bgr, thickness + 2)
    cv2.line(image, (cx, cy + size), (cx, cy + arm_length), border_color_bgr, thickness + 2)
    
    cv2.line(image, (cx - arm_length, cy), (cx - size, cy), fill_color_bgr, thickness)
    cv2.line(image, (cx + size, cy), (cx + arm_length, cy), fill_color_bgr, thickness)
    cv2.line(image, (cx, cy - arm_length), (cx, cy - size), fill_color_bgr, thickness)
    cv2.line(image, (cx, cy + size), (cx, cy + arm_length), fill_color_bgr, thickness)
    
    cv2.circle(image, (cx, cy), size, border_color_bgr, -1)
    cv2.circle(image, (cx, cy), size - 2, fill_color_bgr, -1)
    
    return image


def add_color_marker_to_bb(
    image: np.ndarray,
    bb,
    color_tolerance: int = 40,
    marker_size: int = 8
) -> np.ndarray:
    """
    Add a dominant color crosshair marker to a bounding box if it's medium or large.
    
    Args:
        image: Image to draw on (RGB format, will be modified).
        bb: BoundingBox object.
        color_tolerance: Tolerance for color matching.
        marker_size: Size of the crosshair marker.
    
    Returns:
        Image with marker drawn (or unchanged if box is small).
    """
    if image is None or bb is None:
        return image
    
    analysis = analyze_bounding_box_from_bb(image, bb, color_tolerance)
    
    if analysis is None:
        return image
    
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if len(image.shape) == 3 else image
    
    draw_crosshair_marker(
        image_bgr,
        analysis.center_of_gravity,
        analysis.dominant_color_bgr,
        analysis.contrasting_color_bgr,
        size=marker_size,
        thickness=2
    )
    
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
