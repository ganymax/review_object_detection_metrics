"""Tests for object scale classification and utilities."""

import math
import pytest
import numpy as np

from src.utils.object_scale import (
    ObjectScale,
    COCO_SCALE_THRESHOLDS,
    classify_scale,
    classify_scale_from_dimensions,
    get_scale_color_bgr,
    get_scale_color_rgb,
    get_scale_color_normalized,
    get_scale_color_for_area,
    get_scale_label,
    get_area_range_for_scale,
    ScaleStatistics,
    compute_scale_statistics,
    filter_boxes_by_scale,
    group_boxes_by_scale,
)
from src.bounding_box import BoundingBox
from src.utils.enumerators import BBFormat, BBType, CoordinatesType


class TestCOCOScaleThresholds:
    """Test COCO scale threshold values."""
    
    def test_small_threshold(self):
        assert COCO_SCALE_THRESHOLDS['small_upper'] == 32 ** 2
        assert COCO_SCALE_THRESHOLDS['small_upper'] == 1024
    
    def test_medium_threshold(self):
        assert COCO_SCALE_THRESHOLDS['medium_upper'] == 96 ** 2
        assert COCO_SCALE_THRESHOLDS['medium_upper'] == 9216


class TestClassifyScale:
    """Test scale classification function."""
    
    def test_small_objects(self):
        assert classify_scale(0) == ObjectScale.SMALL
        assert classify_scale(1) == ObjectScale.SMALL
        assert classify_scale(500) == ObjectScale.SMALL
        assert classify_scale(1024) == ObjectScale.SMALL  # Boundary: 32^2
    
    def test_medium_objects(self):
        assert classify_scale(1025) == ObjectScale.MEDIUM
        assert classify_scale(5000) == ObjectScale.MEDIUM
        assert classify_scale(9216) == ObjectScale.MEDIUM  # Boundary: 96^2
    
    def test_large_objects(self):
        assert classify_scale(9217) == ObjectScale.LARGE
        assert classify_scale(10000) == ObjectScale.LARGE
        assert classify_scale(100000) == ObjectScale.LARGE
        assert classify_scale(1000000) == ObjectScale.LARGE
    
    def test_boundary_32_squared(self):
        # Exactly 32^2 should be SMALL
        assert classify_scale(32 ** 2) == ObjectScale.SMALL
        # Just above should be MEDIUM
        assert classify_scale(32 ** 2 + 0.1) == ObjectScale.MEDIUM
    
    def test_boundary_96_squared(self):
        # Exactly 96^2 should be MEDIUM
        assert classify_scale(96 ** 2) == ObjectScale.MEDIUM
        # Just above should be LARGE
        assert classify_scale(96 ** 2 + 0.1) == ObjectScale.LARGE
    
    def test_invalid_inputs(self):
        assert classify_scale(None) == ObjectScale.UNKNOWN
        assert classify_scale(-1) == ObjectScale.UNKNOWN
        assert classify_scale(-100) == ObjectScale.UNKNOWN
        assert classify_scale(float('inf')) == ObjectScale.UNKNOWN
        assert classify_scale(float('-inf')) == ObjectScale.UNKNOWN
        assert classify_scale(float('nan')) == ObjectScale.UNKNOWN
    
    def test_string_conversion(self):
        # Should handle numeric strings
        assert classify_scale("100") == ObjectScale.SMALL
        assert classify_scale("5000") == ObjectScale.MEDIUM
        assert classify_scale("invalid") == ObjectScale.UNKNOWN


class TestClassifyScaleFromDimensions:
    """Test scale classification from width/height."""
    
    def test_small_dimensions(self):
        # 32x32 = 1024 (small boundary)
        assert classify_scale_from_dimensions(32, 32) == ObjectScale.SMALL
        # 20x20 = 400
        assert classify_scale_from_dimensions(20, 20) == ObjectScale.SMALL
    
    def test_medium_dimensions(self):
        # 50x50 = 2500
        assert classify_scale_from_dimensions(50, 50) == ObjectScale.MEDIUM
        # 96x96 = 9216 (medium boundary)
        assert classify_scale_from_dimensions(96, 96) == ObjectScale.MEDIUM
    
    def test_large_dimensions(self):
        # 100x100 = 10000
        assert classify_scale_from_dimensions(100, 100) == ObjectScale.LARGE
        # 200x200 = 40000
        assert classify_scale_from_dimensions(200, 200) == ObjectScale.LARGE
    
    def test_invalid_dimensions(self):
        assert classify_scale_from_dimensions(None, 10) == ObjectScale.UNKNOWN
        assert classify_scale_from_dimensions(10, None) == ObjectScale.UNKNOWN
        assert classify_scale_from_dimensions(-10, 10) == ObjectScale.UNKNOWN
        assert classify_scale_from_dimensions(10, -10) == ObjectScale.UNKNOWN


class TestScaleColors:
    """Test scale color functions."""
    
    def test_bgr_colors(self):
        color = get_scale_color_bgr(ObjectScale.SMALL)
        assert isinstance(color, tuple)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)
    
    def test_rgb_colors(self):
        color = get_scale_color_rgb(ObjectScale.SMALL)
        assert isinstance(color, tuple)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)
    
    def test_normalized_colors(self):
        color = get_scale_color_normalized(ObjectScale.SMALL)
        assert isinstance(color, tuple)
        assert len(color) == 3
        assert all(0.0 <= c <= 1.0 for c in color)
    
    def test_color_for_area(self):
        # Small
        color = get_scale_color_for_area(500, 'rgb')
        assert color == get_scale_color_rgb(ObjectScale.SMALL)
        
        # Medium
        color = get_scale_color_for_area(5000, 'bgr')
        assert color == get_scale_color_bgr(ObjectScale.MEDIUM)
        
        # Large
        color = get_scale_color_for_area(10000, 'normalized')
        assert color == get_scale_color_normalized(ObjectScale.LARGE)
    
    def test_unknown_scale_color(self):
        color = get_scale_color_bgr(ObjectScale.UNKNOWN)
        assert color is not None
        # Should return gray
        assert color == (128, 128, 128)
    
    def test_invalid_scale_type(self):
        # Should return UNKNOWN color for invalid input
        color = get_scale_color_bgr("invalid")
        assert color == get_scale_color_bgr(ObjectScale.UNKNOWN)


class TestScaleLabels:
    """Test scale label functions."""
    
    def test_scale_labels(self):
        assert "32" in get_scale_label(ObjectScale.SMALL)
        assert "96" in get_scale_label(ObjectScale.MEDIUM)
        assert "96" in get_scale_label(ObjectScale.LARGE)
    
    def test_unknown_label(self):
        label = get_scale_label(ObjectScale.UNKNOWN)
        assert "Unknown" in label


class TestAreaRangeForScale:
    """Test area range functions."""
    
    def test_small_range(self):
        min_area, max_area = get_area_range_for_scale(ObjectScale.SMALL)
        assert min_area == 0.0
        assert max_area == 32 ** 2
    
    def test_medium_range(self):
        min_area, max_area = get_area_range_for_scale(ObjectScale.MEDIUM)
        assert min_area == 32 ** 2
        assert max_area == 96 ** 2
    
    def test_large_range(self):
        min_area, max_area = get_area_range_for_scale(ObjectScale.LARGE)
        assert min_area == 96 ** 2
        assert max_area == float('inf')


class TestScaleStatistics:
    """Test ScaleStatistics class."""
    
    def test_empty_statistics(self):
        stats = ScaleStatistics()
        assert stats.total_count == 0
        assert stats.get_count(ObjectScale.SMALL) == 0
        assert stats.get_percentage(ObjectScale.SMALL) == 0.0
    
    def test_add_boxes(self):
        stats = ScaleStatistics()
        
        # Add small boxes
        stats.add_box(500)
        stats.add_box(800)
        
        # Add medium box
        stats.add_box(5000)
        
        # Add large box
        stats.add_box(10000)
        
        assert stats.total_count == 4
        assert stats.get_count(ObjectScale.SMALL) == 2
        assert stats.get_count(ObjectScale.MEDIUM) == 1
        assert stats.get_count(ObjectScale.LARGE) == 1
    
    def test_percentages(self):
        stats = ScaleStatistics()
        stats.add_box(500)
        stats.add_box(5000)
        stats.add_box(10000)
        stats.add_box(20000)
        
        assert stats.get_percentage(ObjectScale.SMALL) == 25.0
        assert stats.get_percentage(ObjectScale.MEDIUM) == 25.0
        assert stats.get_percentage(ObjectScale.LARGE) == 50.0
    
    def test_area_statistics(self):
        stats = ScaleStatistics()
        stats.add_box(100)
        stats.add_box(200)
        stats.add_box(300)
        
        mean = stats.get_mean_area(ObjectScale.SMALL)
        assert mean == 200.0
        
        assert stats.get_min_area(ObjectScale.SMALL) == 100
        assert stats.get_max_area(ObjectScale.SMALL) == 300
    
    def test_summary(self):
        stats = ScaleStatistics()
        stats.add_box(500)
        stats.add_box(5000)
        stats.add_box(10000)
        
        summary = stats.get_summary()
        assert summary['total_count'] == 3
        assert 'by_scale' in summary
        assert 'small' in summary['by_scale']
        assert 'medium' in summary['by_scale']
        assert 'large' in summary['by_scale']
    
    def test_string_representation(self):
        stats = ScaleStatistics()
        stats.add_box(500)
        stats.add_box(5000)
        
        str_repr = str(stats)
        assert 'Scale Statistics' in str_repr
        assert 'Small' in str_repr


class TestBoundingBoxScaleMethods:
    """Test BoundingBox scale-related methods."""
    
    def create_bb(self, x, y, w, h):
        """Helper to create a bounding box."""
        return BoundingBox(
            image_name='test_image',
            class_id='test_class',
            coordinates=(x, y, w, h),
            type_coordinates=CoordinatesType.ABSOLUTE,
            bb_type=BBType.GROUND_TRUTH,
            format=BBFormat.XYWH
        )
    
    def test_get_scale_small(self):
        # 20x20 = 400 pixels (small)
        bb = self.create_bb(0, 0, 20, 20)
        assert bb.get_scale() == ObjectScale.SMALL
        assert bb.is_small() is True
        assert bb.is_medium() is False
        assert bb.is_large() is False
    
    def test_get_scale_medium(self):
        # 50x50 = 2500 pixels (medium)
        bb = self.create_bb(0, 0, 50, 50)
        assert bb.get_scale() == ObjectScale.MEDIUM
        assert bb.is_small() is False
        assert bb.is_medium() is True
        assert bb.is_large() is False
    
    def test_get_scale_large(self):
        # 100x100 = 10000 pixels (large)
        bb = self.create_bb(0, 0, 100, 100)
        assert bb.get_scale() == ObjectScale.LARGE
        assert bb.is_small() is False
        assert bb.is_medium() is False
        assert bb.is_large() is True
    
    def test_get_scale_color(self):
        bb = self.create_bb(0, 0, 20, 20)
        
        rgb_color = bb.get_scale_color('rgb')
        assert isinstance(rgb_color, tuple)
        assert len(rgb_color) == 3
        
        bgr_color = bb.get_scale_color('bgr')
        assert isinstance(bgr_color, tuple)
        
        norm_color = bb.get_scale_color('normalized')
        assert all(0.0 <= c <= 1.0 for c in norm_color)
    
    def test_get_scale_label(self):
        bb = self.create_bb(0, 0, 20, 20)
        label = bb.get_scale_label()
        assert 'Small' in label
    
    def test_coco_area(self):
        # COCO area is (x2-x1)*(y2-y1) without +1
        bb = self.create_bb(0, 0, 10, 10)
        coco_area = bb.get_coco_area()
        assert coco_area == 100  # 10 * 10


class TestStaticScaleMethods:
    """Test static scale-related methods on BoundingBox."""
    
    def create_bbs(self):
        """Create a set of test bounding boxes."""
        bbs = []
        # Small boxes (< 1024 area)
        for _ in range(3):
            bbs.append(BoundingBox(
                image_name='test',
                class_id='cat',
                coordinates=(0, 0, 20, 20),  # 400 area
                type_coordinates=CoordinatesType.ABSOLUTE,
                bb_type=BBType.GROUND_TRUTH,
                format=BBFormat.XYWH
            ))
        # Medium boxes (1024 < area <= 9216)
        for _ in range(2):
            bbs.append(BoundingBox(
                image_name='test',
                class_id='dog',
                coordinates=(0, 0, 50, 50),  # 2500 area
                type_coordinates=CoordinatesType.ABSOLUTE,
                bb_type=BBType.GROUND_TRUTH,
                format=BBFormat.XYWH
            ))
        # Large box (> 9216 area)
        bbs.append(BoundingBox(
            image_name='test',
            class_id='person',
            coordinates=(0, 0, 100, 100),  # 10000 area
            type_coordinates=CoordinatesType.ABSOLUTE,
            bb_type=BBType.GROUND_TRUTH,
            format=BBFormat.XYWH
        ))
        return bbs
    
    def test_get_scale_statistics(self):
        bbs = self.create_bbs()
        stats = BoundingBox.get_scale_statistics(bbs)
        
        assert stats.get_count(ObjectScale.SMALL) == 3
        assert stats.get_count(ObjectScale.MEDIUM) == 2
        assert stats.get_count(ObjectScale.LARGE) == 1
    
    def test_filter_by_scale(self):
        bbs = self.create_bbs()
        
        small = BoundingBox.filter_by_scale(bbs, ObjectScale.SMALL)
        assert len(small) == 3
        
        medium = BoundingBox.filter_by_scale(bbs, ObjectScale.MEDIUM)
        assert len(medium) == 2
        
        large = BoundingBox.filter_by_scale(bbs, ObjectScale.LARGE)
        assert len(large) == 1
    
    def test_group_by_scale(self):
        bbs = self.create_bbs()
        groups = BoundingBox.group_by_scale(bbs)
        
        assert len(groups[ObjectScale.SMALL]) == 3
        assert len(groups[ObjectScale.MEDIUM]) == 2
        assert len(groups[ObjectScale.LARGE]) == 1
    
    def test_get_amount_by_scale(self):
        bbs = self.create_bbs()
        amounts = BoundingBox.get_amount_bounding_box_by_scale(bbs)
        
        assert 'small' in amounts
        assert 'medium' in amounts
        assert 'large' in amounts
        assert amounts['small'] == 3
        assert amounts['medium'] == 2
        assert amounts['large'] == 1


class TestHelperFunctions:
    """Test helper functions."""
    
    def create_bbs(self, sizes):
        """Create bounding boxes with specified sizes."""
        bbs = []
        for w, h in sizes:
            bbs.append(BoundingBox(
                image_name='test',
                class_id='test',
                coordinates=(0, 0, w, h),
                type_coordinates=CoordinatesType.ABSOLUTE,
                bb_type=BBType.GROUND_TRUTH,
                format=BBFormat.XYWH
            ))
        return bbs
    
    def test_compute_scale_statistics(self):
        bbs = self.create_bbs([(20, 20), (50, 50), (100, 100)])
        stats = compute_scale_statistics(bbs)
        
        assert stats.total_count == 3
        assert stats.get_count(ObjectScale.SMALL) == 1
        assert stats.get_count(ObjectScale.MEDIUM) == 1
        assert stats.get_count(ObjectScale.LARGE) == 1
    
    def test_filter_boxes_by_scale(self):
        bbs = self.create_bbs([(20, 20), (20, 20), (50, 50)])
        
        small = filter_boxes_by_scale(bbs, ObjectScale.SMALL)
        assert len(small) == 2
        
        medium = filter_boxes_by_scale(bbs, ObjectScale.MEDIUM)
        assert len(medium) == 1
    
    def test_group_boxes_by_scale(self):
        bbs = self.create_bbs([(20, 20), (50, 50), (100, 100)])
        groups = group_boxes_by_scale(bbs)
        
        assert len(groups[ObjectScale.SMALL]) == 1
        assert len(groups[ObjectScale.MEDIUM]) == 1
        assert len(groups[ObjectScale.LARGE]) == 1
    
    def test_empty_list(self):
        stats = compute_scale_statistics([])
        assert stats.total_count == 0
        
        filtered = filter_boxes_by_scale([], ObjectScale.SMALL)
        assert len(filtered) == 0
        
        groups = group_boxes_by_scale([])
        assert all(len(g) == 0 for g in groups.values())
    
    def test_none_input(self):
        stats = compute_scale_statistics(None)
        assert stats.total_count == 0
        
        filtered = filter_boxes_by_scale(None, ObjectScale.SMALL)
        assert len(filtered) == 0
        
        groups = group_boxes_by_scale(None)
        assert all(len(g) == 0 for g in groups.values())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
