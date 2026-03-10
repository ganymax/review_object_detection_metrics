"""
Object scale classification following COCO standard definitions.

COCO defines object scales based on absolute pixel area:
- Small:  area <= 32^2 (1024 pixels)
- Medium: 32^2 < area <= 96^2 (1024 < area <= 9216 pixels)
- Large:  area > 96^2 (> 9216 pixels)

This module provides:
- Scale classification for bounding boxes
- Color coding based on scale
- Statistical analysis by scale category
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import math


class ObjectScale(Enum):
    """COCO-standard object scale categories based on absolute pixel area."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    UNKNOWN = "unknown"


# COCO standard area thresholds (in pixels squared)
COCO_SCALE_THRESHOLDS = {
    'small_upper': 32 ** 2,      # 1024 pixels
    'medium_upper': 96 ** 2,     # 9216 pixels
}

# BGR colors for OpenCV (reversed RGB for cv2 compatibility)
# Using distinct, colorblind-friendly colors
SCALE_COLORS_BGR = {
    ObjectScale.SMALL: (255, 100, 100),    # Light blue (small objects)
    ObjectScale.MEDIUM: (100, 255, 100),   # Light green (medium objects)
    ObjectScale.LARGE: (100, 100, 255),    # Light red (large objects)
    ObjectScale.UNKNOWN: (128, 128, 128),  # Gray (unknown/invalid)
}

# RGB colors for matplotlib and general use
SCALE_COLORS_RGB = {
    ObjectScale.SMALL: (100, 100, 255),    # Light blue
    ObjectScale.MEDIUM: (100, 255, 100),   # Light green
    ObjectScale.LARGE: (255, 100, 100),    # Light red
    ObjectScale.UNKNOWN: (128, 128, 128),  # Gray
}

# Normalized RGB colors (0-1 range) for matplotlib
SCALE_COLORS_NORMALIZED = {
    ObjectScale.SMALL: (0.39, 0.39, 1.0),
    ObjectScale.MEDIUM: (0.39, 1.0, 0.39),
    ObjectScale.LARGE: (1.0, 0.39, 0.39),
    ObjectScale.UNKNOWN: (0.5, 0.5, 0.5),
}

# Human-readable scale labels
SCALE_LABELS = {
    ObjectScale.SMALL: "Small (≤32²px)",
    ObjectScale.MEDIUM: "Medium (32²-96²px)",
    ObjectScale.LARGE: "Large (>96²px)",
    ObjectScale.UNKNOWN: "Unknown",
}


def classify_scale(area: float) -> ObjectScale:
    """
    Classify an object's scale based on its bounding box area.
    
    Uses COCO standard thresholds:
    - Small:  area <= 32^2 (1024 pixels)
    - Medium: 32^2 < area <= 96^2 (9216 pixels)
    - Large:  area > 96^2
    
    Args:
        area: Bounding box area in absolute pixel values.
              Should be calculated as width * height.
    
    Returns:
        ObjectScale enum value indicating the scale category.
    
    Raises:
        None - returns UNKNOWN for invalid inputs instead of raising.
    """
    if area is None:
        return ObjectScale.UNKNOWN
    
    try:
        area_float = float(area)
    except (TypeError, ValueError):
        return ObjectScale.UNKNOWN
    
    if not math.isfinite(area_float) or area_float < 0:
        return ObjectScale.UNKNOWN
    
    if area_float <= COCO_SCALE_THRESHOLDS['small_upper']:
        return ObjectScale.SMALL
    elif area_float <= COCO_SCALE_THRESHOLDS['medium_upper']:
        return ObjectScale.MEDIUM
    else:
        return ObjectScale.LARGE


def classify_scale_from_dimensions(width: float, height: float) -> ObjectScale:
    """
    Classify scale from width and height dimensions.
    
    Args:
        width: Bounding box width in pixels.
        height: Bounding box height in pixels.
    
    Returns:
        ObjectScale enum value indicating the scale category.
    """
    if width is None or height is None:
        return ObjectScale.UNKNOWN
    
    try:
        w = float(width)
        h = float(height)
    except (TypeError, ValueError):
        return ObjectScale.UNKNOWN
    
    if w < 0 or h < 0 or not math.isfinite(w) or not math.isfinite(h):
        return ObjectScale.UNKNOWN
    
    return classify_scale(w * h)


def get_scale_color_bgr(scale: ObjectScale) -> Tuple[int, int, int]:
    """
    Get BGR color tuple for a given scale (OpenCV format).
    
    Args:
        scale: ObjectScale enum value.
    
    Returns:
        Tuple of (B, G, R) color values (0-255).
    """
    if not isinstance(scale, ObjectScale):
        return SCALE_COLORS_BGR[ObjectScale.UNKNOWN]
    return SCALE_COLORS_BGR.get(scale, SCALE_COLORS_BGR[ObjectScale.UNKNOWN])


def get_scale_color_rgb(scale: ObjectScale) -> Tuple[int, int, int]:
    """
    Get RGB color tuple for a given scale.
    
    Args:
        scale: ObjectScale enum value.
    
    Returns:
        Tuple of (R, G, B) color values (0-255).
    """
    if not isinstance(scale, ObjectScale):
        return SCALE_COLORS_RGB[ObjectScale.UNKNOWN]
    return SCALE_COLORS_RGB.get(scale, SCALE_COLORS_RGB[ObjectScale.UNKNOWN])


def get_scale_color_normalized(scale: ObjectScale) -> Tuple[float, float, float]:
    """
    Get normalized RGB color tuple for a given scale (matplotlib format).
    
    Args:
        scale: ObjectScale enum value.
    
    Returns:
        Tuple of (R, G, B) color values (0.0-1.0).
    """
    if not isinstance(scale, ObjectScale):
        return SCALE_COLORS_NORMALIZED[ObjectScale.UNKNOWN]
    return SCALE_COLORS_NORMALIZED.get(scale, SCALE_COLORS_NORMALIZED[ObjectScale.UNKNOWN])


def get_scale_color_for_area(area: float, color_format: str = 'rgb') -> tuple:
    """
    Get color for a bounding box based on its area.
    
    Args:
        area: Bounding box area in pixels.
        color_format: One of 'rgb', 'bgr', or 'normalized'.
    
    Returns:
        Color tuple in the requested format.
    """
    scale = classify_scale(area)
    
    format_lower = (color_format or 'rgb').lower().strip()
    
    if format_lower == 'bgr':
        return get_scale_color_bgr(scale)
    elif format_lower == 'normalized':
        return get_scale_color_normalized(scale)
    else:
        return get_scale_color_rgb(scale)


def get_scale_label(scale: ObjectScale) -> str:
    """
    Get human-readable label for a scale category.
    
    Args:
        scale: ObjectScale enum value.
    
    Returns:
        Human-readable string label.
    """
    if not isinstance(scale, ObjectScale):
        return SCALE_LABELS[ObjectScale.UNKNOWN]
    return SCALE_LABELS.get(scale, SCALE_LABELS[ObjectScale.UNKNOWN])


def get_area_range_for_scale(scale: ObjectScale) -> Tuple[float, float]:
    """
    Get the area range (min, max) for a given scale category.
    
    Args:
        scale: ObjectScale enum value.
    
    Returns:
        Tuple of (min_area, max_area). Uses float('inf') for unbounded upper limit.
    """
    if scale == ObjectScale.SMALL:
        return (0.0, COCO_SCALE_THRESHOLDS['small_upper'])
    elif scale == ObjectScale.MEDIUM:
        return (COCO_SCALE_THRESHOLDS['small_upper'], COCO_SCALE_THRESHOLDS['medium_upper'])
    elif scale == ObjectScale.LARGE:
        return (COCO_SCALE_THRESHOLDS['medium_upper'], float('inf'))
    else:
        return (0.0, float('inf'))


class ScaleStatistics:
    """
    Compute and store statistics about bounding boxes grouped by scale.
    """
    
    def __init__(self):
        self._counts = {scale: 0 for scale in ObjectScale}
        self._areas = {scale: [] for scale in ObjectScale}
        self._total_count = 0
    
    def add_box(self, area: float) -> ObjectScale:
        """
        Add a bounding box's area to the statistics.
        
        Args:
            area: Bounding box area in pixels.
        
        Returns:
            The ObjectScale category the box was classified into.
        """
        scale = classify_scale(area)
        self._counts[scale] += 1
        self._total_count += 1
        
        if area is not None and math.isfinite(float(area)) and float(area) >= 0:
            self._areas[scale].append(float(area))
        
        return scale
    
    def add_box_from_dimensions(self, width: float, height: float) -> ObjectScale:
        """
        Add a bounding box by its dimensions.
        
        Args:
            width: Box width in pixels.
            height: Box height in pixels.
        
        Returns:
            The ObjectScale category the box was classified into.
        """
        if width is None or height is None:
            return self.add_box(None)
        
        try:
            area = float(width) * float(height)
        except (TypeError, ValueError):
            area = None
        
        return self.add_box(area)
    
    @property
    def counts(self) -> Dict[ObjectScale, int]:
        """Get counts per scale category."""
        return dict(self._counts)
    
    @property
    def total_count(self) -> int:
        """Get total number of boxes processed."""
        return self._total_count
    
    def get_count(self, scale: ObjectScale) -> int:
        """Get count for a specific scale."""
        return self._counts.get(scale, 0)
    
    def get_percentage(self, scale: ObjectScale) -> float:
        """Get percentage of boxes in a scale category."""
        if self._total_count == 0:
            return 0.0
        return (self._counts.get(scale, 0) / self._total_count) * 100.0
    
    def get_mean_area(self, scale: ObjectScale) -> Optional[float]:
        """Get mean area for a scale category."""
        areas = self._areas.get(scale, [])
        if not areas:
            return None
        return sum(areas) / len(areas)
    
    def get_min_area(self, scale: ObjectScale) -> Optional[float]:
        """Get minimum area for a scale category."""
        areas = self._areas.get(scale, [])
        if not areas:
            return None
        return min(areas)
    
    def get_max_area(self, scale: ObjectScale) -> Optional[float]:
        """Get maximum area for a scale category."""
        areas = self._areas.get(scale, [])
        if not areas:
            return None
        return max(areas)
    
    def get_std_area(self, scale: ObjectScale) -> Optional[float]:
        """Get standard deviation of areas for a scale category."""
        areas = self._areas.get(scale, [])
        if len(areas) < 2:
            return None
        
        mean = sum(areas) / len(areas)
        variance = sum((x - mean) ** 2 for x in areas) / len(areas)
        return math.sqrt(variance)
    
    def get_summary(self) -> Dict[str, any]:
        """
        Get a comprehensive summary of scale statistics.
        
        Returns:
            Dictionary containing counts, percentages, and area statistics
            for each scale category.
        """
        summary = {
            'total_count': self._total_count,
            'by_scale': {}
        }
        
        for scale in [ObjectScale.SMALL, ObjectScale.MEDIUM, ObjectScale.LARGE]:
            scale_data = {
                'count': self.get_count(scale),
                'percentage': self.get_percentage(scale),
                'mean_area': self.get_mean_area(scale),
                'min_area': self.get_min_area(scale),
                'max_area': self.get_max_area(scale),
                'std_area': self.get_std_area(scale),
                'label': get_scale_label(scale),
            }
            summary['by_scale'][scale.value] = scale_data
        
        # Add unknown if any exist
        if self._counts[ObjectScale.UNKNOWN] > 0:
            summary['by_scale']['unknown'] = {
                'count': self.get_count(ObjectScale.UNKNOWN),
                'percentage': self.get_percentage(ObjectScale.UNKNOWN),
            }
        
        return summary
    
    def __str__(self) -> str:
        """String representation of scale statistics."""
        lines = [f"Scale Statistics (Total: {self._total_count})"]
        lines.append("-" * 50)
        
        for scale in [ObjectScale.SMALL, ObjectScale.MEDIUM, ObjectScale.LARGE]:
            count = self.get_count(scale)
            pct = self.get_percentage(scale)
            mean = self.get_mean_area(scale)
            
            mean_str = f"{mean:.1f}" if mean is not None else "N/A"
            lines.append(f"  {get_scale_label(scale):20s}: {count:5d} ({pct:5.1f}%) - Mean area: {mean_str}")
        
        if self._counts[ObjectScale.UNKNOWN] > 0:
            count = self.get_count(ObjectScale.UNKNOWN)
            pct = self.get_percentage(ObjectScale.UNKNOWN)
            lines.append(f"  {'Unknown':20s}: {count:5d} ({pct:5.1f}%)")
        
        return "\n".join(lines)


def compute_scale_statistics(bounding_boxes: List) -> ScaleStatistics:
    """
    Compute scale statistics for a list of BoundingBox objects.
    
    Args:
        bounding_boxes: List of BoundingBox objects.
    
    Returns:
        ScaleStatistics object with computed statistics.
    """
    stats = ScaleStatistics()
    
    if bounding_boxes is None:
        return stats
    
    for bb in bounding_boxes:
        if bb is None:
            continue
        
        try:
            area = bb.get_area()
            stats.add_box(area)
        except (AttributeError, TypeError):
            # Handle case where bb doesn't have get_area() or it fails
            try:
                coords = bb.get_absolute_bounding_box()
                if coords and len(coords) >= 4:
                    w = coords[2] if len(coords) == 4 else coords[2] - coords[0]
                    h = coords[3] if len(coords) == 4 else coords[3] - coords[1]
                    stats.add_box_from_dimensions(w, h)
                else:
                    stats.add_box(None)
            except Exception:
                stats.add_box(None)
    
    return stats


def filter_boxes_by_scale(
    bounding_boxes: List,
    scale: ObjectScale,
    include_unknown: bool = False
) -> List:
    """
    Filter bounding boxes to only include those of a specific scale.
    
    Args:
        bounding_boxes: List of BoundingBox objects.
        scale: ObjectScale to filter for.
        include_unknown: If True, include boxes with unknown scale when filtering.
    
    Returns:
        Filtered list of BoundingBox objects.
    """
    if bounding_boxes is None:
        return []
    
    filtered = []
    
    for bb in bounding_boxes:
        if bb is None:
            continue
        
        try:
            area = bb.get_area()
            bb_scale = classify_scale(area)
        except Exception:
            bb_scale = ObjectScale.UNKNOWN
        
        if bb_scale == scale:
            filtered.append(bb)
        elif include_unknown and bb_scale == ObjectScale.UNKNOWN:
            filtered.append(bb)
    
    return filtered


def group_boxes_by_scale(bounding_boxes: List) -> Dict[ObjectScale, List]:
    """
    Group bounding boxes by their scale category.
    
    Args:
        bounding_boxes: List of BoundingBox objects.
    
    Returns:
        Dictionary mapping ObjectScale to lists of BoundingBox objects.
    """
    groups = {scale: [] for scale in ObjectScale}
    
    if bounding_boxes is None:
        return groups
    
    for bb in bounding_boxes:
        if bb is None:
            continue
        
        try:
            area = bb.get_area()
            scale = classify_scale(area)
        except Exception:
            scale = ObjectScale.UNKNOWN
        
        groups[scale].append(bb)
    
    return groups
