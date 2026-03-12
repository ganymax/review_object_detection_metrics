"""Tests for color analysis utilities (bias profiling)."""

import numpy as np
import pytest

from src.utils.color_analysis import (
    ColorAnalysisResult,
    quantize_color,
    extract_dominant_color,
    create_color_mask,
    calculate_center_of_gravity,
    analyze_bounding_box_color,
    should_analyze_bounding_box,
    get_contrasting_border_color,
)
from src.utils.object_scale import ObjectScale, COCO_SCALE_THRESHOLDS


class TestQuantizeColor:
    """Test color quantization function."""
    
    def test_basic_quantization(self):
        pixel = np.array([100, 150, 200])
        result = quantize_color(pixel, levels=8)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(0 <= c <= 255 for c in result)
    
    def test_quantization_levels(self):
        pixel = np.array([127, 127, 127])
        # With 8 levels, step = 32, so 127 // 32 * 32 + 16 = 112
        result = quantize_color(pixel, levels=8)
        assert result == (112, 112, 112)
    
    def test_boundary_values(self):
        # Test with max values
        pixel = np.array([255, 255, 255])
        result = quantize_color(pixel, levels=8)
        assert all(c <= 255 for c in result)
        
        # Test with min values
        pixel = np.array([0, 0, 0])
        result = quantize_color(pixel, levels=8)
        assert all(c >= 0 for c in result)


class TestExtractDominantColor:
    """Test dominant color extraction."""
    
    def test_uniform_color_roi(self):
        # Create a 10x10 red ROI
        roi = np.full((10, 10, 3), [255, 0, 0], dtype=np.uint8)
        result = extract_dominant_color(roi)
        # Should be close to red (with quantization)
        assert result[0] > 200  # R channel high
        assert result[1] < 50   # G channel low
        assert result[2] < 50   # B channel low
    
    def test_mixed_color_roi_majority_wins(self):
        # Create ROI with 70% blue, 30% green
        roi = np.zeros((10, 10, 3), dtype=np.uint8)
        roi[:7, :, :] = [0, 0, 255]  # Blue (70%)
        roi[7:, :, :] = [0, 255, 0]  # Green (30%)
        
        result = extract_dominant_color(roi)
        # Blue should dominate
        assert result[2] > result[1]  # B > G
    
    def test_empty_roi_raises(self):
        roi = np.array([]).reshape(0, 0, 3)
        with pytest.raises(ValueError):
            extract_dominant_color(roi)
    
    def test_none_roi_raises(self):
        with pytest.raises(ValueError):
            extract_dominant_color(None)
    
    def test_wrong_shape_raises(self):
        roi = np.zeros((10, 10), dtype=np.uint8)  # 2D instead of 3D
        with pytest.raises(ValueError):
            extract_dominant_color(roi)
    
    def test_single_pixel_roi(self):
        roi = np.array([[[128, 64, 32]]], dtype=np.uint8)
        result = extract_dominant_color(roi)
        assert isinstance(result, tuple)
        assert len(result) == 3


class TestCreateColorMask:
    """Test color mask creation."""
    
    def test_exact_match(self):
        roi = np.full((5, 5, 3), [100, 150, 200], dtype=np.uint8)
        mask = create_color_mask(roi, (100, 150, 200), tolerance=0)
        assert mask.shape == (5, 5)
        assert np.all(mask)  # All pixels should match
    
    def test_within_tolerance(self):
        roi = np.full((5, 5, 3), [100, 150, 200], dtype=np.uint8)
        mask = create_color_mask(roi, (110, 160, 210), tolerance=20)
        assert np.all(mask)  # Should match within tolerance
    
    def test_outside_tolerance(self):
        roi = np.full((5, 5, 3), [100, 150, 200], dtype=np.uint8)
        mask = create_color_mask(roi, (200, 50, 100), tolerance=10)
        assert not np.any(mask)  # No pixels should match
    
    def test_partial_match(self):
        roi = np.zeros((10, 10, 3), dtype=np.uint8)
        roi[:5, :, :] = [255, 0, 0]  # Top half red
        roi[5:, :, :] = [0, 0, 255]  # Bottom half blue
        
        mask = create_color_mask(roi, (255, 0, 0), tolerance=10)
        assert np.sum(mask) == 50  # 5x10 = 50 matching pixels
    
    def test_empty_roi(self):
        roi = np.array([]).reshape(0, 0, 3)
        mask = create_color_mask(roi, (100, 100, 100))
        assert mask.size == 0


class TestCalculateCenterOfGravity:
    """Test center of gravity calculation."""
    
    def test_uniform_mask(self):
        mask = np.ones((10, 10), dtype=bool)
        cog = calculate_center_of_gravity(mask)
        assert cog is not None
        # Center should be at (4.5, 4.5) for 0-indexed 10x10 grid
        assert abs(cog[0] - 4.5) < 0.01
        assert abs(cog[1] - 4.5) < 0.01
    
    def test_single_pixel_mask(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[3, 7] = True
        cog = calculate_center_of_gravity(mask)
        assert cog == (7, 3)  # (x, y) format
    
    def test_corner_weighted_mask(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[0, 0] = True
        cog = calculate_center_of_gravity(mask)
        assert cog == (0, 0)
    
    def test_empty_mask(self):
        mask = np.zeros((10, 10), dtype=bool)
        cog = calculate_center_of_gravity(mask)
        assert cog is None
    
    def test_none_mask(self):
        cog = calculate_center_of_gravity(None)
        assert cog is None
    
    def test_asymmetric_distribution(self):
        mask = np.zeros((10, 10), dtype=bool)
        # Put more weight on the right side
        mask[:, 8:] = True  # Last 2 columns
        cog = calculate_center_of_gravity(mask)
        # X should be around 8.5 (average of 8 and 9)
        assert cog[0] > 7


class TestAnalyzeBoundingBoxColor:
    """Test full bounding box color analysis."""
    
    def test_basic_analysis(self):
        # Create a simple 100x100 image with a red region
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[20:40, 30:50, :] = [255, 0, 0]  # Red box
        
        result = analyze_bounding_box_color(image, (30, 20, 50, 40))
        
        assert result is not None
        assert isinstance(result, ColorAnalysisResult)
        assert result.dominant_color_rgb[0] > 200  # Red dominant
        assert result.roi_pixel_count == 20 * 20
    
    def test_center_of_gravity_calculation(self):
        # Create image with uniform color in bbox
        image = np.full((100, 100, 3), [100, 100, 100], dtype=np.uint8)
        
        result = analyze_bounding_box_color(image, (10, 10, 50, 50))
        
        assert result is not None
        # COG should be near center of ROI
        roi_cx = (50 - 10) / 2  # 20
        roi_cy = (50 - 10) / 2  # 20
        assert abs(result.center_of_gravity[0] - roi_cx) < 5
        assert abs(result.center_of_gravity[1] - roi_cy) < 5
    
    def test_absolute_coordinates(self):
        image = np.full((100, 100, 3), [128, 128, 128], dtype=np.uint8)
        
        result = analyze_bounding_box_color(image, (20, 30, 60, 70))
        
        assert result is not None
        # Absolute COG should be offset by bbox position
        abs_x, abs_y = result.center_of_gravity_absolute
        assert 20 <= abs_x <= 60
        assert 30 <= abs_y <= 70
    
    def test_bbox_clamping(self):
        image = np.full((50, 50, 3), [100, 100, 100], dtype=np.uint8)
        
        # Bbox extends beyond image
        result = analyze_bounding_box_color(image, (-10, -10, 100, 100))
        
        assert result is not None
        # Should be clamped to image bounds
        assert result.roi_pixel_count == 50 * 50
    
    def test_empty_bbox(self):
        image = np.full((100, 100, 3), [100, 100, 100], dtype=np.uint8)
        
        # Zero-size bbox
        result = analyze_bounding_box_color(image, (50, 50, 50, 50))
        
        # Should handle gracefully (either None or valid result with minimal ROI)
        # Implementation clamps to at least 1 pixel
        assert result is None or result.roi_pixel_count >= 1


class TestShouldAnalyzeBoundingBox:
    """Test scale-based filtering for analysis."""
    
    def test_small_box_excluded(self):
        # Small: area <= 32^2 = 1024
        assert should_analyze_bounding_box(500) is False
        assert should_analyze_bounding_box(1024) is False
    
    def test_medium_box_included(self):
        # Medium: 32^2 < area <= 96^2
        assert should_analyze_bounding_box(1025) is True
        assert should_analyze_bounding_box(5000) is True
        assert should_analyze_bounding_box(9216) is True
    
    def test_large_box_included(self):
        # Large: area > 96^2
        assert should_analyze_bounding_box(9217) is True
        assert should_analyze_bounding_box(50000) is True
    
    def test_boundary_values(self):
        small_upper = COCO_SCALE_THRESHOLDS['small_upper']
        medium_upper = COCO_SCALE_THRESHOLDS['medium_upper']
        
        assert should_analyze_bounding_box(small_upper) is False
        assert should_analyze_bounding_box(small_upper + 1) is True
        assert should_analyze_bounding_box(medium_upper) is True
        assert should_analyze_bounding_box(medium_upper + 1) is True


class TestGetContrastingBorderColor:
    """Test contrasting border color selection."""
    
    def test_dark_color_gets_white_border(self):
        dark_colors = [
            (0, 0, 0),       # Black
            (50, 50, 50),    # Dark gray
            (0, 0, 128),     # Dark blue
            (128, 0, 0),     # Dark red
        ]
        for color in dark_colors:
            border = get_contrasting_border_color(color)
            assert border == (255, 255, 255), f"Dark color {color} should get white border"
    
    def test_light_color_gets_black_border(self):
        light_colors = [
            (255, 255, 255),  # White
            (200, 200, 200),  # Light gray
            (255, 255, 0),    # Yellow
            (255, 200, 200),  # Light pink
        ]
        for color in light_colors:
            border = get_contrasting_border_color(color)
            assert border == (0, 0, 0), f"Light color {color} should get black border"
    
    def test_green_weighted_luminance(self):
        # Green has highest luminance weight (0.587)
        # So (0, 255, 0) should be considered light
        border = get_contrasting_border_color((0, 255, 0))
        assert border == (0, 0, 0)  # Black border for bright green
    
    def test_mid_gray(self):
        # (127, 127, 127) has luminance ~127, at threshold
        border = get_contrasting_border_color((127, 127, 127))
        # Should get white (luminance < 128)
        assert border == (255, 255, 255)


class TestColorAnalysisIntegration:
    """Integration tests for the full color analysis pipeline."""
    
    def test_gradient_image(self):
        # Create image with horizontal gradient
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        for x in range(100):
            image[:, x, 0] = x * 2  # R increases left to right
        
        result = analyze_bounding_box_color(image, (0, 0, 100, 100))
        
        assert result is not None
        # With quantization, dominant color will be in a specific bin
        # The COG will be at pixels matching that quantized color
        # Just verify we get a valid result
        assert 0 <= result.center_of_gravity[0] <= 100
        assert 0 <= result.center_of_gravity[1] <= 100
    
    def test_checkerboard_pattern(self):
        # Create checkerboard with red and blue
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        for y in range(100):
            for x in range(100):
                if (x + y) % 2 == 0:
                    image[y, x] = [255, 0, 0]  # Red
                else:
                    image[y, x] = [0, 0, 255]  # Blue
        
        result = analyze_bounding_box_color(image, (0, 0, 100, 100))
        
        assert result is not None
        # Both colors equally represented, COG should be near center
        assert abs(result.center_of_gravity[0] - 50) < 10
        assert abs(result.center_of_gravity[1] - 50) < 10
    
    def test_real_world_scenario(self):
        # Simulate a detection with object in corner
        image = np.full((200, 200, 3), [50, 100, 50], dtype=np.uint8)  # Green background
        # Add bright object in bottom-right of bbox region
        image[60:80, 80:100, :] = [255, 200, 100]  # Orange-ish object
        
        result = analyze_bounding_box_color(image, (50, 50, 100, 100))
        
        assert result is not None
        # Should detect the object's color or background depending on coverage


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_very_small_roi(self):
        image = np.full((100, 100, 3), [128, 128, 128], dtype=np.uint8)
        result = analyze_bounding_box_color(image, (50, 50, 52, 52))
        # 2x2 ROI should still work
        assert result is not None
        assert result.roi_pixel_count == 4
    
    def test_single_channel_handling(self):
        # The function expects 3-channel, but test graceful handling
        roi = np.zeros((10, 10, 3), dtype=np.uint8)
        roi[:, :, 0] = 255  # Only red channel
        result = extract_dominant_color(roi)
        assert result[0] > 200  # Red should dominate
    
    def test_float_coordinates(self):
        image = np.full((100, 100, 3), [100, 100, 100], dtype=np.uint8)
        # Float coordinates should be handled
        result = analyze_bounding_box_color(image, (10.5, 20.7, 50.2, 60.9))
        assert result is not None
    
    def test_negative_coordinates(self):
        image = np.full((100, 100, 3), [100, 100, 100], dtype=np.uint8)
        result = analyze_bounding_box_color(image, (-10, -10, 50, 50))
        assert result is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
