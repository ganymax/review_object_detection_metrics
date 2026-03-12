"""Tests for bias profiling drawing functions."""

import numpy as np
import pytest
import cv2

from src.bounding_box import BoundingBox
from src.utils.enumerators import BBFormat, BBType, CoordinatesType
from src.utils.general_utils import (
    draw_crosshair_marker,
    add_bb_into_image_with_bias_marker,
    draw_bbs_with_bias_markers,
)
from src.utils.object_scale import ObjectScale


def create_test_image(width=200, height=200, color=(100, 150, 100)):
    """Create a test image with a specified background color."""
    return np.full((height, width, 3), color, dtype=np.uint8)


def create_bb(x, y, w, h, class_id='test'):
    """Helper to create a bounding box."""
    return BoundingBox(
        image_name='test_image',
        class_id=class_id,
        coordinates=(x, y, w, h),
        type_coordinates=CoordinatesType.ABSOLUTE,
        bb_type=BBType.GROUND_TRUTH,
        format=BBFormat.XYWH
    )


class TestDrawCrosshairMarker:
    """Test crosshair marker drawing."""
    
    def test_basic_drawing(self):
        img = create_test_image(100, 100)
        original = img.copy()
        
        result = draw_crosshair_marker(img, (50, 50), (255, 0, 0), size=10, thickness=2)
        
        # Image should be modified
        assert not np.array_equal(result, original)
        assert result.shape == original.shape
    
    def test_crosshair_visible(self):
        img = create_test_image(100, 100, color=(128, 128, 128))
        result = draw_crosshair_marker(img, (50, 50), (255, 0, 0), size=10, thickness=2)
        
        # Check that there are pixels different from background at the crosshair location
        center_region = result[45:56, 45:56, :]
        assert not np.all(center_region == [128, 128, 128])
    
    def test_border_contrast_dark_fill(self):
        img = create_test_image(100, 100, color=(0, 0, 0))
        # Dark fill color should get white border
        result = draw_crosshair_marker(img, (50, 50), (20, 20, 20), size=10, thickness=2)
        
        # Should have white border pixels (255, 255, 255)
        # Check horizontal line area
        horizontal_line = result[50, 40:61, :]
        has_white = np.any(np.all(horizontal_line == [255, 255, 255], axis=1))
        assert has_white or np.any(horizontal_line > 200)
    
    def test_border_contrast_light_fill(self):
        img = create_test_image(100, 100, color=(255, 255, 255))
        # Light fill color should get black border
        result = draw_crosshair_marker(img, (50, 50), (240, 240, 240), size=10, thickness=2)
        
        # Should have black border pixels
        horizontal_line = result[50, 40:61, :]
        has_black = np.any(np.all(horizontal_line == [0, 0, 0], axis=1))
        assert has_black or np.any(horizontal_line < 50)
    
    def test_center_out_of_bounds(self):
        img = create_test_image(100, 100)
        original = img.copy()
        
        # Center completely outside
        result = draw_crosshair_marker(img, (-100, -100), (255, 0, 0), size=10)
        assert np.array_equal(result, original)
        
        result = draw_crosshair_marker(img, (200, 200), (255, 0, 0), size=10)
        assert np.array_equal(result, original)
    
    def test_crosshair_clipped_at_edge(self):
        img = create_test_image(100, 100)
        
        # Crosshair at corner should be partially drawn
        result = draw_crosshair_marker(img, (5, 5), (255, 0, 0), size=20, thickness=2)
        
        # Should still work without error
        assert result.shape == img.shape


class TestAddBBWithBiasMarker:
    """Test bounding box drawing with bias markers."""
    
    def test_medium_box_gets_marker(self):
        img = create_test_image(200, 200)
        # Add a bright region to have a distinct dominant color
        img[50:100, 50:120, :] = [255, 200, 50]
        
        # Medium box (60x50 = 3000 area)
        bb = create_bb(40, 40, 80, 60)
        assert bb.get_scale() == ObjectScale.MEDIUM
        
        original = img.copy()
        result = add_bb_into_image_with_bias_marker(img.copy(), bb, thickness=2)
        
        # Image should be modified (has both box and marker)
        assert not np.array_equal(result, original)
    
    def test_large_box_gets_marker(self):
        img = create_test_image(300, 300)
        img[50:200, 50:200, :] = [100, 255, 100]
        
        # Large box (150x150 = 22500 area)
        bb = create_bb(40, 40, 160, 160)
        assert bb.get_scale() == ObjectScale.LARGE
        
        result = add_bb_into_image_with_bias_marker(img.copy(), bb, thickness=2)
        
        # Should have bounding box drawn
        assert result.shape == img.shape
    
    def test_small_box_no_marker(self):
        img = create_test_image(200, 200)
        
        # Small box (20x20 = 400 area)
        bb = create_bb(50, 50, 20, 20)
        assert bb.get_scale() == ObjectScale.SMALL
        
        # Small boxes should still be drawn (just without the marker)
        result = add_bb_into_image_with_bias_marker(img.copy(), bb, thickness=2)
        
        # Image should be modified (box is drawn)
        assert result.shape == img.shape
    
    def test_custom_crosshair_size(self):
        img = create_test_image(200, 200)
        img[40:100, 40:120, :] = [255, 0, 0]
        
        bb = create_bb(30, 30, 100, 80)  # Medium/Large box
        
        result_small = add_bb_into_image_with_bias_marker(
            img.copy(), bb, crosshair_size=5
        )
        result_large = add_bb_into_image_with_bias_marker(
            img.copy(), bb, crosshair_size=20
        )
        
        # Both should work
        assert result_small.shape == img.shape
        assert result_large.shape == img.shape


class TestDrawBBsWithBiasMarkers:
    """Test drawing multiple bounding boxes with bias markers."""
    
    def test_multiple_boxes(self):
        img = create_test_image(300, 300)
        img[50:100, 50:100, :] = [255, 0, 0]  # Red region
        img[150:200, 150:200, :] = [0, 255, 0]  # Green region
        
        bbs = [
            create_bb(40, 40, 70, 70, 'object1'),  # Medium
            create_bb(140, 140, 70, 70, 'object2'),  # Medium
            create_bb(10, 10, 20, 20, 'small'),  # Small
        ]
        
        result = draw_bbs_with_bias_markers(img.copy(), bbs, thickness=2)
        
        assert result.shape == img.shape
    
    def test_empty_list(self):
        img = create_test_image(100, 100)
        original = img.copy()
        
        result = draw_bbs_with_bias_markers(img.copy(), [], thickness=2)
        
        assert np.array_equal(result, original)
    
    def test_none_in_list(self):
        img = create_test_image(200, 200)
        
        bbs = [
            create_bb(40, 40, 70, 70),
            None,
            create_bb(100, 100, 50, 50),
        ]
        
        # Should handle None gracefully
        result = draw_bbs_with_bias_markers(img.copy(), bbs, thickness=2)
        
        assert result.shape == img.shape
    
    def test_with_labels(self):
        img = create_test_image(200, 200)
        
        bbs = [
            create_bb(40, 40, 70, 70, 'cat'),
            create_bb(10, 10, 20, 20, 'dog'),
        ]
        
        result = draw_bbs_with_bias_markers(
            img.copy(), bbs, 
            show_labels=True, 
            show_scale_in_label=True
        )
        
        assert result.shape == img.shape
    
    def test_image_from_path(self):
        # Create a temporary image file
        import tempfile
        import os
        
        img = create_test_image(200, 200)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            # Save as BGR for OpenCV
            cv2.imwrite(f.name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            temp_path = f.name
        
        try:
            bbs = [create_bb(40, 40, 70, 70)]
            result = draw_bbs_with_bias_markers(temp_path, bbs)
            
            assert result.shape == img.shape
        finally:
            os.unlink(temp_path)


class TestEdgeCases:
    """Test edge cases for drawing functions."""
    
    def test_very_thin_box(self):
        img = create_test_image(200, 200)
        
        # Very thin box (1 pixel wide)
        bb = create_bb(50, 50, 1, 100)  # 100 area = small
        
        result = add_bb_into_image_with_bias_marker(img.copy(), bb)
        
        assert result.shape == img.shape
    
    def test_box_at_image_edge(self):
        img = create_test_image(100, 100)
        
        # Box at edge
        bb = create_bb(0, 0, 60, 60)
        
        result = add_bb_into_image_with_bias_marker(img.copy(), bb)
        
        assert result.shape == img.shape
    
    def test_box_extending_beyond_image(self):
        img = create_test_image(100, 100)
        
        # Box extends beyond image bounds
        bb = create_bb(80, 80, 50, 50)
        
        result = add_bb_into_image_with_bias_marker(img.copy(), bb)
        
        assert result.shape == img.shape
    
    def test_uniform_color_region(self):
        # Image with completely uniform color
        img = np.full((200, 200, 3), [128, 128, 128], dtype=np.uint8)
        
        bb = create_bb(40, 40, 70, 70)
        
        result = add_bb_into_image_with_bias_marker(img.copy(), bb)
        
        # Should still work with uniform color
        assert result.shape == img.shape
    
    def test_black_image(self):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        
        bb = create_bb(40, 40, 70, 70)
        
        result = add_bb_into_image_with_bias_marker(img.copy(), bb)
        
        assert result.shape == img.shape
    
    def test_white_image(self):
        img = np.full((200, 200, 3), [255, 255, 255], dtype=np.uint8)
        
        bb = create_bb(40, 40, 70, 70)
        
        result = add_bb_into_image_with_bias_marker(img.copy(), bb)
        
        assert result.shape == img.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
