"""Integration tests for scale-based metrics and visualization."""

import os
import sys
import pytest
import numpy as np

from src.bounding_box import BoundingBox
from src.utils.enumerators import BBFormat, BBType, CoordinatesType, ObjectScale
from src.utils.object_scale import (
    classify_scale,
    compute_scale_statistics,
    get_scale_label,
    ScaleStatistics,
)
import src.evaluators.coco_evaluator as coco_evaluator
import src.utils.general_utils as general_utils


def create_test_bboxes():
    """Create a diverse set of test bounding boxes."""
    bbs = []
    
    # Small objects (area <= 1024)
    # 30x30 = 900
    for i in range(5):
        bbs.append(BoundingBox(
            image_name=f'img_{i}',
            class_id='small_obj',
            coordinates=(10, 10, 30, 30),
            type_coordinates=CoordinatesType.ABSOLUTE,
            bb_type=BBType.GROUND_TRUTH,
            format=BBFormat.XYWH
        ))
    
    # Medium objects (1024 < area <= 9216)
    # 60x60 = 3600
    for i in range(3):
        bbs.append(BoundingBox(
            image_name=f'img_{i + 5}',
            class_id='medium_obj',
            coordinates=(50, 50, 60, 60),
            type_coordinates=CoordinatesType.ABSOLUTE,
            bb_type=BBType.GROUND_TRUTH,
            format=BBFormat.XYWH
        ))
    
    # Large objects (area > 9216)
    # 100x100 = 10000
    for i in range(2):
        bbs.append(BoundingBox(
            image_name=f'img_{i + 8}',
            class_id='large_obj',
            coordinates=(100, 100, 100, 100),
            type_coordinates=CoordinatesType.ABSOLUTE,
            bb_type=BBType.GROUND_TRUTH,
            format=BBFormat.XYWH
        ))
    
    return bbs


def create_detection_bboxes():
    """Create detection bounding boxes matching the GT pattern."""
    dets = []
    
    # Detections for small objects
    for i in range(4):  # 4 out of 5 detected
        dets.append(BoundingBox(
            image_name=f'img_{i}',
            class_id='small_obj',
            coordinates=(12, 12, 28, 28),  # Slightly offset
            type_coordinates=CoordinatesType.ABSOLUTE,
            bb_type=BBType.DETECTED,
            confidence=0.9 - i * 0.1,
            format=BBFormat.XYWH
        ))
    
    # Detections for medium objects
    for i in range(3):
        dets.append(BoundingBox(
            image_name=f'img_{i + 5}',
            class_id='medium_obj',
            coordinates=(52, 52, 58, 58),
            type_coordinates=CoordinatesType.ABSOLUTE,
            bb_type=BBType.DETECTED,
            confidence=0.85 - i * 0.1,
            format=BBFormat.XYWH
        ))
    
    # Detections for large objects
    for i in range(2):
        dets.append(BoundingBox(
            image_name=f'img_{i + 8}',
            class_id='large_obj',
            coordinates=(105, 105, 95, 95),
            type_coordinates=CoordinatesType.ABSOLUTE,
            bb_type=BBType.DETECTED,
            confidence=0.95 - i * 0.1,
            format=BBFormat.XYWH
        ))
    
    # Add a false positive
    dets.append(BoundingBox(
        image_name='img_0',
        class_id='small_obj',
        coordinates=(200, 200, 25, 25),
        type_coordinates=CoordinatesType.ABSOLUTE,
        bb_type=BBType.DETECTED,
        confidence=0.3,
        format=BBFormat.XYWH
    ))
    
    return dets


class TestScaleIntegration:
    """Test scale integration with existing functionality."""
    
    def test_bounding_box_scale_methods(self):
        bbs = create_test_bboxes()
        
        small_bbs = [bb for bb in bbs if bb.is_small()]
        medium_bbs = [bb for bb in bbs if bb.is_medium()]
        large_bbs = [bb for bb in bbs if bb.is_large()]
        
        assert len(small_bbs) == 5
        assert len(medium_bbs) == 3
        assert len(large_bbs) == 2
    
    def test_scale_statistics(self):
        bbs = create_test_bboxes()
        stats = BoundingBox.get_scale_statistics(bbs)
        
        assert stats.total_count == 10
        assert stats.get_count(ObjectScale.SMALL) == 5
        assert stats.get_count(ObjectScale.MEDIUM) == 3
        assert stats.get_count(ObjectScale.LARGE) == 2
        
        assert stats.get_percentage(ObjectScale.SMALL) == 50.0
        assert stats.get_percentage(ObjectScale.MEDIUM) == 30.0
        assert stats.get_percentage(ObjectScale.LARGE) == 20.0
    
    def test_group_by_scale(self):
        bbs = create_test_bboxes()
        groups = BoundingBox.group_by_scale(bbs)
        
        assert len(groups[ObjectScale.SMALL]) == 5
        assert len(groups[ObjectScale.MEDIUM]) == 3
        assert len(groups[ObjectScale.LARGE]) == 2
    
    def test_filter_by_scale(self):
        bbs = create_test_bboxes()
        
        small = BoundingBox.filter_by_scale(bbs, ObjectScale.SMALL)
        assert len(small) == 5
        assert all(bb.is_small() for bb in small)
        
        medium = BoundingBox.filter_by_scale(bbs, ObjectScale.MEDIUM)
        assert len(medium) == 3
        
        large = BoundingBox.filter_by_scale(bbs, ObjectScale.LARGE)
        assert len(large) == 2
    
    def test_scale_colors(self):
        bbs = create_test_bboxes()
        
        for bb in bbs:
            rgb = bb.get_scale_color('rgb')
            bgr = bb.get_scale_color('bgr')
            norm = bb.get_scale_color('normalized')
            
            assert len(rgb) == 3
            assert len(bgr) == 3
            assert len(norm) == 3
            
            assert all(0 <= c <= 255 for c in rgb)
            assert all(0 <= c <= 255 for c in bgr)
            assert all(0.0 <= c <= 1.0 for c in norm)


class TestCOCOEvaluatorScaleMetrics:
    """Test COCO evaluator scale-based metrics."""
    
    def test_get_scale_distribution(self):
        bbs = create_test_bboxes()
        dist = coco_evaluator.get_scale_distribution(bbs)
        
        assert dist['total_count'] == 10
        assert 'by_scale' in dist
        assert 'small' in dist['by_scale']
        assert 'medium' in dist['by_scale']
        assert 'large' in dist['by_scale']
        
        assert dist['by_scale']['small']['count'] == 5
        assert dist['by_scale']['medium']['count'] == 3
        assert dist['by_scale']['large']['count'] == 2
    
    def test_get_coco_summary_with_scale_details(self):
        gt_bbs = create_test_bboxes()
        det_bbs = create_detection_bboxes()
        
        summary = coco_evaluator.get_coco_summary_with_scale_details(gt_bbs, det_bbs)
        
        # Standard COCO metrics should be present
        assert 'AP' in summary
        assert 'AP50' in summary
        assert 'AP75' in summary
        assert 'APsmall' in summary
        assert 'APmedium' in summary
        assert 'APlarge' in summary
        
        # Scale distributions should be present
        assert 'gt_scale_distribution' in summary
        assert 'det_scale_distribution' in summary
        assert 'scale_metrics' in summary
        
        # Check GT distribution
        gt_dist = summary['gt_scale_distribution']
        assert gt_dist['total_count'] == 10
        
        # Check detection distribution
        det_dist = summary['det_scale_distribution']
        assert det_dist['total_count'] == 10  # 9 matches + 1 FP
    
    def test_get_coco_metrics_by_scale(self):
        gt_bbs = create_test_bboxes()
        det_bbs = create_detection_bboxes()
        
        small_metrics = coco_evaluator.get_coco_metrics_by_scale(
            gt_bbs, det_bbs, ObjectScale.SMALL
        )
        medium_metrics = coco_evaluator.get_coco_metrics_by_scale(
            gt_bbs, det_bbs, ObjectScale.MEDIUM
        )
        large_metrics = coco_evaluator.get_coco_metrics_by_scale(
            gt_bbs, det_bbs, ObjectScale.LARGE
        )
        
        # Each should return metrics dict
        assert isinstance(small_metrics, dict)
        assert isinstance(medium_metrics, dict)
        assert isinstance(large_metrics, dict)
    
    def test_format_scale_metrics_report(self):
        gt_bbs = create_test_bboxes()
        det_bbs = create_detection_bboxes()
        
        report = coco_evaluator.format_scale_metrics_report(gt_bbs, det_bbs)
        
        assert isinstance(report, str)
        assert 'COCO EVALUATION REPORT' in report
        assert 'SCALE ANALYSIS' in report
        assert 'Small' in report
        assert 'Medium' in report
        assert 'Large' in report
        assert 'Precision' in report or 'Prec' in report


class TestVisualizationUtils:
    """Test visualization utility functions."""
    
    def test_plot_bb_per_scale(self):
        bbs = create_test_bboxes()
        
        # Should not raise
        plt = general_utils.plot_bb_per_scale(bbs, show=False)
        assert plt is not None
        plt.close()
    
    def test_plot_scale_comparison(self):
        gt_bbs = create_test_bboxes()
        det_bbs = create_detection_bboxes()
        
        # Should not raise
        plt = general_utils.plot_scale_comparison(gt_bbs, det_bbs, show=False)
        assert plt is not None
        plt.close()
    
    def test_create_scale_legend_image(self):
        legend = general_utils.create_scale_legend_image()
        
        assert legend is not None
        assert legend.shape[2] == 3  # BGR
        assert legend.shape[0] > 0
        assert legend.shape[1] > 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_bboxes(self):
        stats = BoundingBox.get_scale_statistics([])
        assert stats.total_count == 0
        
        groups = BoundingBox.group_by_scale([])
        assert all(len(g) == 0 for g in groups.values())
        
        filtered = BoundingBox.filter_by_scale([], ObjectScale.SMALL)
        assert len(filtered) == 0
    
    def test_none_input(self):
        stats = compute_scale_statistics(None)
        assert stats.total_count == 0
    
    def test_very_small_box(self):
        bb = BoundingBox(
            image_name='test',
            class_id='tiny',
            coordinates=(0, 0, 1, 1),
            type_coordinates=CoordinatesType.ABSOLUTE,
            bb_type=BBType.GROUND_TRUTH,
            format=BBFormat.XYWH
        )
        assert bb.get_scale() == ObjectScale.SMALL
    
    def test_very_large_box(self):
        bb = BoundingBox(
            image_name='test',
            class_id='huge',
            coordinates=(0, 0, 1000, 1000),
            type_coordinates=CoordinatesType.ABSOLUTE,
            bb_type=BBType.GROUND_TRUTH,
            format=BBFormat.XYWH
        )
        assert bb.get_scale() == ObjectScale.LARGE
    
    def test_boundary_boxes(self):
        # Exactly at 32^2 = 1024 boundary
        bb_32 = BoundingBox(
            image_name='test',
            class_id='boundary',
            coordinates=(0, 0, 32, 32),
            type_coordinates=CoordinatesType.ABSOLUTE,
            bb_type=BBType.GROUND_TRUTH,
            format=BBFormat.XYWH
        )
        assert bb_32.get_scale() == ObjectScale.SMALL
        
        # Exactly at 96^2 = 9216 boundary
        bb_96 = BoundingBox(
            image_name='test',
            class_id='boundary',
            coordinates=(0, 0, 96, 96),
            type_coordinates=CoordinatesType.ABSOLUTE,
            bb_type=BBType.GROUND_TRUTH,
            format=BBFormat.XYWH
        )
        assert bb_96.get_scale() == ObjectScale.MEDIUM


class TestScaleMetricsAccuracy:
    """Test accuracy of scale-based metrics."""
    
    def test_scale_metrics_structure(self):
        gt_bbs = create_test_bboxes()
        det_bbs = create_detection_bboxes()
        
        summary = coco_evaluator.get_coco_summary_with_scale_details(gt_bbs, det_bbs)
        scale_metrics = summary.get('scale_metrics', {})
        
        for scale_name in ['small', 'medium', 'large']:
            if scale_name in scale_metrics:
                m = scale_metrics[scale_name]
                assert 'gt_count' in m
                assert 'det_count' in m
                assert 'precision' in m
                assert 'recall' in m
                assert 'f1_score' in m
                assert 'TP' in m
                assert 'FP' in m
                assert 'FN' in m
                
                # Validate metric ranges
                assert 0.0 <= m['precision'] <= 1.0
                assert 0.0 <= m['recall'] <= 1.0
                assert 0.0 <= m['f1_score'] <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
