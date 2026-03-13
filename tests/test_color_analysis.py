"""Tests for color analysis utilities for bias profiling."""

import numpy as np
import pytest
import cv2

from src.utils.color_analysis import (
    ColorAnalysisResult,
    extract_roi,
    find_dominant_color,
    create_color_mask,
    calculate_center_of_gravity,
    analyze_bounding_box_color,
    analyze_bounding_box_from_bb,
    draw_crosshair_marker,
    add_color_marker_to_bb,
)
from src.utils.object_scale import ObjectScale
from src.bounding_box import BoundingBox
from src.utils.enumerators import BBFormat, BBType, CoordinatesType


class TestColorAnalysisResult:
    """Test ColorAnalysisResult dataclass."""
    
    def test_basic_creation(self):
        result = ColorAnalysisResult(
            dominant_color_rgb=(255, 0, 0),
            center_of_gravity=(50, 50),
            mask_pixel_count=100,
            roi_bounds=(0, 0, 100, 100),
            scale=ObjectScale.LARGE
        )
        assert result.dominant_color_rgb == (255, 0, 0)
        assert result.center_of_gravity == (50, 50)
        assert result.mask_pixel_count == 100
        assert result.scale == ObjectScale.LARGE
    
    def test_bgr_conversion(self):
        result = ColorAnalysisResult(
            dominant_color_rgb=(255, 128, 64),
            center_of_gravity=(0, 0),
            mask_pixel_count=0,
            roi_bounds=(0, 0, 10, 10),
            scale=ObjectScale.MEDIUM
        )
        assert result.dominant_color_bgr == (64, 128, 255)
    
    def test_contrasting_color_dark_dominant(self):
        # Dark color should contrast with white
        result = ColorAnalysisResult(
            dominant_color_rgb=(30, 30, 30),
            center_of_gravity=(0, 0),
            mask_pixel_count=0,
            roi_bounds=(0, 0, 10, 10),
            scale=ObjectScale.MEDIUM
        )
        assert result.contrasting_color_rgb == (255, 255, 255)
    
    def test_contrasting_color_light_dominant(self):
        # Light color should contrast with black
        result = ColorAnalysisResult(
            dominant_color_rgb=(200, 200, 200),
            center_of_gravity=(0, 0),
            mask_pixel_count=0,
            roi_bounds=(0, 0, 10, 10),
            scale=ObjectScale.MEDIUM
        )
        assert result.contrasting_color_rgb == (0, 0, 0)


class TestExtractROI:
    """Test ROI extraction function."""
    
    def test_valid_extraction(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[20:40, 30:60] = (255, 0, 0)
        
        roi = extract_roi(image, (30, 20, 60, 40))
        assert roi is not None
        assert roi.shape == (20, 30, 3)
    
    def test_boundary_clipping(self):
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        roi = extract_roi(image, (-10, -10, 60, 60))
        assert roi is not None
        assert roi.shape == (50, 50, 3)
    
    def test_invalid_bounds(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        roi = extract_roi(image, (50, 50, 40, 40))  # x2 < x1
        assert roi is None
    
    def test_none_image(self):
        roi = extract_roi(None, (0, 0, 10, 10))
        assert roi is None


class TestFindDominantColor:
    """Test dominant color finding."""
    
    def test_solid_color_region(self):
        roi = np.full((50, 50, 3), (255, 0, 0), dtype=np.uint8)
        color = find_dominant_color(roi)
        assert color[0] > 200  # Red channel should be dominant
        assert color[1] < 50
        assert color[2] < 50
    
    def test_mixed_color_region(self):
        roi = np.zeros((100, 100, 3), dtype=np.uint8)
        roi[:50, :] = (255, 0, 0)  # Top half red
        roi[50:, :] = (0, 0, 255)  # Bottom half blue
        
        color = find_dominant_color(roi)
        # Should return one of the two colors (whichever cluster is larger)
        assert color is not None
        assert len(color) == 3
    
    def test_empty_roi(self):
        roi = np.array([])
        color = find_dominant_color(roi)
        assert color == (128, 128, 128)  # Default gray
    
    def test_grayscale_roi(self):
        roi = np.full((30, 30), 128, dtype=np.uint8)
        color = find_dominant_color(roi)
        assert len(color) == 3
    
    def test_small_roi(self):
        roi = np.full((3, 3, 3), (100, 150, 200), dtype=np.uint8)
        color = find_dominant_color(roi)
        assert abs(color[0] - 100) < 10
        assert abs(color[1] - 150) < 10
        assert abs(color[2] - 200) < 10


class TestCreateColorMask:
    """Test color mask creation."""
    
    def test_exact_color_match(self):
        roi = np.full((50, 50, 3), (100, 150, 200), dtype=np.uint8)
        mask = create_color_mask(roi, (100, 150, 200), tolerance=10)
        # Most pixels should match
        assert np.sum(mask > 0) > 0.9 * 50 * 50
    
    def test_no_color_match(self):
        roi = np.full((50, 50, 3), (255, 0, 0), dtype=np.uint8)
        mask = create_color_mask(roi, (0, 255, 0), tolerance=10)
        # Very few pixels should match
        assert np.sum(mask > 0) < 0.1 * 50 * 50
    
    def test_tolerance_effect(self):
        roi = np.full((50, 50, 3), (100, 100, 100), dtype=np.uint8)
        
        tight_mask = create_color_mask(roi, (110, 110, 110), tolerance=10)
        loose_mask = create_color_mask(roi, (110, 110, 110), tolerance=50)
        
        assert np.sum(loose_mask > 0) >= np.sum(tight_mask > 0)
    
    def test_empty_roi(self):
        roi = np.array([])
        mask = create_color_mask(roi, (100, 100, 100))
        assert mask.shape == (1, 1)


class TestCalculateCenterOfGravity:
    """Test center of gravity calculation."""
    
    def test_centered_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255
        
        cx, cy = calculate_center_of_gravity(mask)
        assert abs(cx - 50) < 5
        assert abs(cy - 50) < 5
    
    def test_corner_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:20, 0:20] = 255
        
        cx, cy = calculate_center_of_gravity(mask)
        assert cx < 20
        assert cy < 20
    
    def test_empty_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        cx, cy = calculate_center_of_gravity(mask)
        # Should return center of image
        assert cx == 50
        assert cy == 50
    
    def test_single_pixel_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30, 70] = 255
        
        cx, cy = calculate_center_of_gravity(mask)
        assert abs(cx - 70) < 2
        assert abs(cy - 30) < 2


class TestAnalyzeBoundingBoxColor:
    """Test complete bounding box color analysis."""
    
    def test_large_box_analysis(self):
        # Large box: area > 96^2 = 9216
        image = np.full((200, 200, 3), (50, 100, 150), dtype=np.uint8)
        
        result = analyze_bounding_box_color(image, (0, 0, 100, 100))
        
        assert result is not None
        assert result.scale == ObjectScale.LARGE
        assert result.center_of_gravity is not None
    
    def test_medium_box_analysis(self):
        # Medium box: 32^2 < area <= 96^2
        image = np.full((100, 100, 3), (200, 50, 100), dtype=np.uint8)
        
        result = analyze_bounding_box_color(image, (0, 0, 50, 50))
        
        assert result is not None
        assert result.scale == ObjectScale.MEDIUM
    
    def test_small_box_skipped(self):
        # Small box: area <= 32^2 = 1024
        image = np.full((100, 100, 3), (100, 100, 100), dtype=np.uint8)
        
        result = analyze_bounding_box_color(image, (0, 0, 30, 30))
        
        # Should return None for small boxes
        assert result is None
    
    def test_invalid_image(self):
        result = analyze_bounding_box_color(None, (0, 0, 100, 100))
        assert result is None


class TestAnalyzeBoundingBoxFromBB:
    """Test analysis using BoundingBox objects."""
    
    def create_bb(self, x, y, w, h):
        return BoundingBox(
            image_name='test',
            class_id='test',
            coordinates=(x, y, w, h),
            type_coordinates=CoordinatesType.ABSOLUTE,
            bb_type=BBType.GROUND_TRUTH,
            format=BBFormat.XYWH
        )
    
    def test_large_bb_analysis(self):
        image = np.full((200, 200, 3), (80, 120, 200), dtype=np.uint8)
        bb = self.create_bb(0, 0, 150, 150)
        
        result = analyze_bounding_box_from_bb(image, bb)
        
        assert result is not None
        assert result.scale == ObjectScale.LARGE
    
    def test_small_bb_skipped(self):
        image = np.full((100, 100, 3), (100, 100, 100), dtype=np.uint8)
        bb = self.create_bb(0, 0, 20, 20)
        
        result = analyze_bounding_box_from_bb(image, bb)
        assert result is None
    
    def test_none_bb(self):
        image = np.full((100, 100, 3), (100, 100, 100), dtype=np.uint8)
        result = analyze_bounding_box_from_bb(image, None)
        assert result is None


class TestDrawCrosshairMarker:
    """Test crosshair marker drawing."""
    
    def test_marker_drawn(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        original_sum = np.sum(image)
        
        draw_crosshair_marker(
            image,
            center=(50, 50),
            fill_color_bgr=(255, 0, 0),
            border_color_bgr=(255, 255, 255),
            size=10,
            thickness=2
        )
        
        # Image should have changed
        assert np.sum(image) > original_sum
    
    def test_marker_at_edge(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Should handle edge positions gracefully
        draw_crosshair_marker(
            image,
            center=(5, 5),
            fill_color_bgr=(0, 255, 0),
            border_color_bgr=(0, 0, 0),
            size=10
        )
        
        # Should not crash and should draw something
        assert np.sum(image) > 0
    
    def test_none_image(self):
        result = draw_crosshair_marker(
            None,
            center=(50, 50),
            fill_color_bgr=(255, 0, 0),
            border_color_bgr=(0, 0, 0)
        )
        assert result is None


class TestAddColorMarkerToBB:
    """Test adding color marker to bounding box."""
    
    def create_bb(self, x, y, w, h):
        return BoundingBox(
            image_name='test',
            class_id='test',
            coordinates=(x, y, w, h),
            type_coordinates=CoordinatesType.ABSOLUTE,
            bb_type=BBType.GROUND_TRUTH,
            format=BBFormat.XYWH
        )
    
    def test_marker_added_to_large_bb(self):
        image = np.full((200, 200, 3), (100, 150, 200), dtype=np.uint8)
        bb = self.create_bb(0, 0, 150, 150)
        
        original_sum = np.sum(image)
        result = add_color_marker_to_bb(image.copy(), bb)
        
        # Image should have changed (marker added)
        assert np.sum(result) != original_sum
    
    def test_no_marker_for_small_bb(self):
        image = np.full((100, 100, 3), (100, 150, 200), dtype=np.uint8)
        bb = self.create_bb(0, 0, 20, 20)
        
        original = image.copy()
        result = add_color_marker_to_bb(image.copy(), bb)
        
        # Image should be unchanged
        np.testing.assert_array_equal(result, original)


class TestIntegration:
    """Integration tests for the full workflow."""
    
    def create_test_image_with_regions(self):
        """Create a test image with distinct colored regions."""
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        
        # Red region in top-left
        image[50:100, 50:100] = (255, 0, 0)
        
        # Green region in center
        image[125:175, 125:175] = (0, 255, 0)
        
        # Blue region in bottom-right
        image[200:250, 200:250] = (0, 0, 255)
        
        return image
    
    def test_full_analysis_workflow(self):
        image = self.create_test_image_with_regions()
        
        # Analyze a region containing mostly red
        result = analyze_bounding_box_color(image, (40, 40, 110, 110))
        
        assert result is not None
        assert result.scale in (ObjectScale.MEDIUM, ObjectScale.LARGE)
        # Red should be dominant
        assert result.dominant_color_rgb[0] > 100
    
    def test_cog_in_correct_quadrant(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        # Put bright pixels in top-left corner with larger area
        image[5:60, 5:60] = (200, 200, 200)
        
        result = analyze_bounding_box_color(image, (0, 0, 100, 100))
        
        assert result is not None
        # Center of gravity should be in top-left region (allowing some tolerance)
        cx, cy = result.center_of_gravity
        assert cx < 60  # Should be biased toward top-left
        assert cy < 60


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_very_small_roi(self):
        image = np.full((10, 10, 3), (128, 128, 128), dtype=np.uint8)
        # This is a small box, should return None
        result = analyze_bounding_box_color(image, (0, 0, 5, 5))
        assert result is None
    
    def test_single_color_image(self):
        image = np.full((200, 200, 3), (100, 100, 100), dtype=np.uint8)
        result = analyze_bounding_box_color(image, (0, 0, 150, 150))
        
        assert result is not None
        # Should return the single color
        for c in result.dominant_color_rgb:
            assert abs(c - 100) < 20
    
    def test_high_contrast_image(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[:100, :] = (255, 255, 255)  # Top half white
        image[100:, :] = (0, 0, 0)  # Bottom half black
        
        result = analyze_bounding_box_color(image, (0, 0, 200, 200))
        
        assert result is not None
        # Contrasting color should be opposite
        lum = sum(result.dominant_color_rgb) / 3
        contrast_lum = sum(result.contrasting_color_rgb) / 3
        assert abs(lum - contrast_lum) > 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
