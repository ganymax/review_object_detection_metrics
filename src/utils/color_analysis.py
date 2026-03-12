"""
Color analysis utilities for dataset bias profiling.

This module provides functions to:
- Extract the dominant RGB color from a bounding box region of interest (ROI)
- Calculate the center of gravity of pixels matching the dominant color
- Support bias analysis by examining color and spatial properties of detections

Only Medium and Large scale bounding boxes are analyzed (Small boxes are ignored).
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from src.utils.object_scale import ObjectScale, classify_scale


@dataclass
class ColorAnalysisResult:
    """Result of color analysis for a bounding box ROI."""
    dominant_color_rgb: Tuple[int, int, int]
    center_of_gravity: Tuple[float, float]  # (x, y) relative to ROI top-left
    center_of_gravity_absolute: Tuple[float, float]  # (x, y) in image coordinates
    mask_pixel_count: int
    roi_pixel_count: int
    coverage_ratio: float


def quantize_color(pixel: np.ndarray, levels: int = 8) -> Tuple[int, int, int]:
    """
    Quantize a pixel color to reduce the color space.
    
    Args:
        pixel: RGB pixel values as numpy array.
        levels: Number of quantization levels per channel (default 8 gives 512 colors).
    
    Returns:
        Quantized RGB tuple.
    """
    step = 256 // levels
    quantized = (pixel // step) * step + step // 2
    quantized = np.clip(quantized, 0, 255)
    return tuple(quantized.astype(int))


def extract_dominant_color(
    roi: np.ndarray,
    quantization_levels: int = 8,
    min_coverage: float = 0.01
) -> Tuple[int, int, int]:
    """
    Extract the single most dominant RGB color from an ROI.
    
    Uses color quantization and histogram counting to find the most frequent color.
    
    Args:
        roi: Region of interest as numpy array (H, W, 3) in RGB format.
        quantization_levels: Number of quantization levels per channel.
        min_coverage: Minimum coverage ratio required (not used for selection,
                      but could be used for filtering in future).
    
    Returns:
        Dominant color as (R, G, B) tuple with values 0-255.
    
    Raises:
        ValueError: If ROI is empty or has invalid shape.
    """
    if roi is None or roi.size == 0:
        raise ValueError("ROI is empty or None")
    
    if len(roi.shape) != 3 or roi.shape[2] != 3:
        raise ValueError(f"ROI must have shape (H, W, 3), got {roi.shape}")
    
    height, width = roi.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("ROI has zero dimensions")
    
    # Reshape to (N, 3) array of pixels
    pixels = roi.reshape(-1, 3)
    
    # Quantize all pixels
    step = 256 // quantization_levels
    quantized = (pixels // step) * step + step // 2
    quantized = np.clip(quantized, 0, 255).astype(np.uint8)
    
    # Convert to hashable representation for counting
    # Pack RGB into single integer: R*256^2 + G*256 + B
    packed = (quantized[:, 0].astype(np.int32) * 65536 + 
              quantized[:, 1].astype(np.int32) * 256 + 
              quantized[:, 2].astype(np.int32))
    
    # Count occurrences
    unique, counts = np.unique(packed, return_counts=True)
    
    # Find the most frequent color
    max_idx = np.argmax(counts)
    dominant_packed = unique[max_idx]
    
    # Unpack back to RGB
    r = (dominant_packed >> 16) & 0xFF
    g = (dominant_packed >> 8) & 0xFF
    b = dominant_packed & 0xFF
    
    return (int(r), int(g), int(b))


def create_color_mask(
    roi: np.ndarray,
    target_color: Tuple[int, int, int],
    tolerance: int = 32
) -> np.ndarray:
    """
    Create a binary mask of pixels matching the target color within tolerance.
    
    Args:
        roi: Region of interest as numpy array (H, W, 3) in RGB format.
        target_color: Target RGB color tuple.
        tolerance: Maximum distance per channel to consider a match.
    
    Returns:
        Binary mask as numpy array (H, W) with True for matching pixels.
    """
    if roi is None or roi.size == 0:
        return np.array([], dtype=bool).reshape(0, 0)
    
    target = np.array(target_color, dtype=np.int32)
    roi_int = roi.astype(np.int32)
    
    # Check each channel is within tolerance
    diff = np.abs(roi_int - target)
    mask = np.all(diff <= tolerance, axis=2)
    
    return mask


def calculate_center_of_gravity(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Calculate the center of gravity (centroid) of True pixels in a binary mask.
    
    Args:
        mask: Binary mask as numpy array (H, W).
    
    Returns:
        (x, y) coordinates of the center of gravity, or None if mask is empty.
        Coordinates are relative to the mask's top-left corner.
    """
    if mask is None or mask.size == 0:
        return None
    
    # Get coordinates of all True pixels
    y_coords, x_coords = np.where(mask)
    
    if len(x_coords) == 0:
        return None
    
    # Calculate centroid
    cx = float(np.mean(x_coords))
    cy = float(np.mean(y_coords))
    
    return (cx, cy)


def analyze_bounding_box_color(
    image: np.ndarray,
    bbox_coords: Tuple[int, int, int, int],
    quantization_levels: int = 8,
    color_tolerance: int = 32
) -> Optional[ColorAnalysisResult]:
    """
    Analyze the dominant color and its spatial distribution within a bounding box.
    
    Args:
        image: Full image as numpy array (H, W, 3) in RGB format.
        bbox_coords: Bounding box as (x1, y1, x2, y2) in absolute pixel coordinates.
        quantization_levels: Number of color quantization levels.
        color_tolerance: Tolerance for color matching when creating mask.
    
    Returns:
        ColorAnalysisResult with dominant color and center of gravity,
        or None if analysis fails (e.g., empty ROI).
    """
    x1, y1, x2, y2 = bbox_coords
    
    # Ensure coordinates are valid integers
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # Clamp to image bounds
    img_h, img_w = image.shape[:2]
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(x1 + 1, min(x2, img_w))
    y2 = max(y1 + 1, min(y2, img_h))
    
    # Extract ROI
    roi = image[y1:y2, x1:x2]
    
    if roi.size == 0:
        return None
    
    roi_h, roi_w = roi.shape[:2]
    roi_pixel_count = roi_h * roi_w
    
    try:
        # Find dominant color
        dominant_color = extract_dominant_color(roi, quantization_levels)
        
        # Create mask for dominant color
        mask = create_color_mask(roi, dominant_color, color_tolerance)
        
        # Calculate center of gravity
        cog_relative = calculate_center_of_gravity(mask)
        
        if cog_relative is None:
            # Fallback to ROI center if mask is empty
            cog_relative = (roi_w / 2.0, roi_h / 2.0)
            mask_pixel_count = 0
        else:
            mask_pixel_count = int(np.sum(mask))
        
        # Convert to absolute image coordinates
        cog_absolute = (x1 + cog_relative[0], y1 + cog_relative[1])
        
        coverage_ratio = mask_pixel_count / roi_pixel_count if roi_pixel_count > 0 else 0.0
        
        return ColorAnalysisResult(
            dominant_color_rgb=dominant_color,
            center_of_gravity=cog_relative,
            center_of_gravity_absolute=cog_absolute,
            mask_pixel_count=mask_pixel_count,
            roi_pixel_count=roi_pixel_count,
            coverage_ratio=coverage_ratio
        )
        
    except (ValueError, IndexError):
        return None


def should_analyze_bounding_box(area: float) -> bool:
    """
    Determine if a bounding box should be analyzed based on its scale.
    
    Only Medium and Large boxes are analyzed; Small boxes are ignored.
    
    Args:
        area: Bounding box area in pixels.
    
    Returns:
        True if the box should be analyzed, False otherwise.
    """
    scale = classify_scale(area)
    return scale in (ObjectScale.MEDIUM, ObjectScale.LARGE)


def get_contrasting_border_color(
    fill_color: Tuple[int, int, int]
) -> Tuple[int, int, int]:
    """
    Get a contrasting border color (black or white) for a given fill color.
    
    Uses luminance calculation to determine if black or white provides better contrast.
    
    Args:
        fill_color: RGB fill color tuple.
    
    Returns:
        Either (0, 0, 0) for black or (255, 255, 255) for white.
    """
    r, g, b = fill_color
    # ITU-R BT.601 luminance formula
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Use white border for dark colors, black for light colors
    if luminance < 128:
        return (255, 255, 255)
    else:
        return (0, 0, 0)
