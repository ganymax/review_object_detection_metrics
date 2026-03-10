""" version ported from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

    Notes:
        1) The default area thresholds here follows the values defined in COCO, that is,
        small:           area <= 32**2
        medium: 32**2 <= area <= 96**2
        large:  96**2 <= area.
        If area is not specified, all areas are considered.

        2) COCO's ground truths contain an 'area' attribute that is associated with the segmented area if
        segmentation-level information exists. While coco uses this 'area' attribute to distinguish between
        'small', 'medium', and 'large' objects, this implementation simply uses the associated bounding box
        area to filter the ground truths.

        3) COCO uses floating point bounding boxes, thus, the calculation of the box area
        for IoU purposes is the simple open-ended delta (x2 - x1) * (y2 - y1).
        PASCALVOC uses integer-based bounding boxes, and the area includes the outer edge,
        that is, (x2 - x1 + 1) * (y2 - y1 + 1). This implementation assumes the open-ended (former)
        convention for area calculation.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from src.bounding_box import BBFormat
from src.utils.object_scale import (
    ObjectScale,
    COCO_SCALE_THRESHOLDS,
    classify_scale,
    get_scale_label,
    get_area_range_for_scale,
    ScaleStatistics,
    compute_scale_statistics,
    group_boxes_by_scale,
)


def get_coco_summary(groundtruth_bbs, detected_bbs):
    """Calculate the 12 standard metrics used in COCOEval,
        AP, AP50, AP75,
        AR1, AR10, AR100,
        APsmall, APmedium, APlarge,
        ARsmall, ARmedium, ARlarge.

        When no ground-truth can be associated with a particular class (NPOS == 0),
        that class is removed from the average calculation.
        If for a given calculation, no metrics whatsoever are available, returns NaN.

    Parameters
        ----------
            groundtruth_bbs : list
                A list containing objects of type BoundingBox representing the ground-truth bounding boxes.
            detected_bbs : list
                A list containing objects of type BoundingBox representing the detected bounding boxes.
    Returns:
            A dictionary with one entry for each metric.
    """

    # separate bbs per image X class
    _bbs = _group_detections(detected_bbs, groundtruth_bbs)

    # pairwise ious
    _ious = {k: _compute_ious(**v) for k, v in _bbs.items()}

    def _evaluate(iou_threshold, max_dets, area_range):
        # accumulate evaluations on a per-class basis
        _evals = defaultdict(lambda: {"scores": [], "matched": [], "NP": []})
        for img_id, class_id in _bbs:
            ev = _evaluate_image(
                _bbs[img_id, class_id]["dt"],
                _bbs[img_id, class_id]["gt"],
                _ious[img_id, class_id],
                iou_threshold,
                max_dets,
                area_range,
            )
            acc = _evals[class_id]
            acc["scores"].append(ev["scores"])
            acc["matched"].append(ev["matched"])
            acc["NP"].append(ev["NP"])

        # now reduce accumulations
        for class_id in _evals:
            acc = _evals[class_id]
            acc["scores"] = np.concatenate(acc["scores"])
            acc["matched"] = np.concatenate(acc["matched"]).astype(bool)
            acc["NP"] = np.sum(acc["NP"])

        res = []
        # run ap calculation per-class
        for class_id in _evals:
            ev = _evals[class_id]
            res.append({
                "class": class_id,
                **_compute_ap_recall(ev["scores"], ev["matched"], ev["NP"]),
            })
        return res

    iou_thresholds = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)

    # compute simple AP with all thresholds, using up to 100 dets, and all areas
    full = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(0, np.inf))
        for i in iou_thresholds
    }

    AP50 = np.mean([x['AP'] for x in full[0.50] if x['AP'] is not None])
    AP75 = np.mean([x['AP'] for x in full[0.75] if x['AP'] is not None])
    AP = np.mean([x['AP'] for k in full for x in full[k] if x['AP'] is not None])

    # max recall for 100 dets can also be calculated here
    AR100 = np.mean(
        [x['TP'] / x['total positives'] for k in full for x in full[k] if x['TP'] is not None])

    small = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(0, 32**2))
        for i in iou_thresholds
    }
    APsmall = [x['AP'] for k in small for x in small[k] if x['AP'] is not None]
    APsmall = np.nan if APsmall == [] else np.mean(APsmall)
    ARsmall = [
        x['TP'] / x['total positives'] for k in small for x in small[k] if x['TP'] is not None
    ]
    ARsmall = np.nan if ARsmall == [] else np.mean(ARsmall)

    medium = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(32**2, 96**2))
        for i in iou_thresholds
    }
    APmedium = [x['AP'] for k in medium for x in medium[k] if x['AP'] is not None]
    APmedium = np.nan if APmedium == [] else np.mean(APmedium)
    ARmedium = [
        x['TP'] / x['total positives'] for k in medium for x in medium[k] if x['TP'] is not None
    ]
    ARmedium = np.nan if ARmedium == [] else np.mean(ARmedium)

    large = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(96**2, np.inf))
        for i in iou_thresholds
    }
    APlarge = [x['AP'] for k in large for x in large[k] if x['AP'] is not None]
    APlarge = np.nan if APlarge == [] else np.mean(APlarge)
    ARlarge = [
        x['TP'] / x['total positives'] for k in large for x in large[k] if x['TP'] is not None
    ]
    ARlarge = np.nan if ARlarge == [] else np.mean(ARlarge)

    max_det1 = {
        i: _evaluate(iou_threshold=i, max_dets=1, area_range=(0, np.inf))
        for i in iou_thresholds
    }
    AR1 = np.mean([
        x['TP'] / x['total positives'] for k in max_det1 for x in max_det1[k] if x['TP'] is not None
    ])

    max_det10 = {
        i: _evaluate(iou_threshold=i, max_dets=10, area_range=(0, np.inf))
        for i in iou_thresholds
    }
    AR10 = np.mean([
        x['TP'] / x['total positives'] for k in max_det10 for x in max_det10[k]
        if x['TP'] is not None
    ])

    return {
        "AP": AP,
        "AP50": AP50,
        "AP75": AP75,
        "APsmall": APsmall,
        "APmedium": APmedium,
        "APlarge": APlarge,
        "AR1": AR1,
        "AR10": AR10,
        "AR100": AR100,
        "ARsmall": ARsmall,
        "ARmedium": ARmedium,
        "ARlarge": ARlarge
    }


def get_coco_metrics(
        groundtruth_bbs,
        detected_bbs,
        iou_threshold=0.5,
        area_range=(0, np.inf),
        max_dets=100,
):
    """ Calculate the Average Precision and Recall metrics as in COCO's official implementation
        given an IOU threshold, area range and maximum number of detections.
    Parameters
        ----------
            groundtruth_bbs : list
                A list containing objects of type BoundingBox representing the ground-truth bounding boxes.
            detected_bbs : list
                A list containing objects of type BoundingBox representing the detected bounding boxes.
            iou_threshold : float
                Intersection Over Union (IOU) value used to consider a TP detection.
            area_range : (numerical x numerical)
                Lower and upper bounds on annotation areas that should be considered.
            max_dets : int
                Upper bound on the number of detections to be considered for each class in an image.

    Returns:
            A list of dictionaries. One dictionary for each class.
            The keys of each dictionary are:
            dict['class']: class representing the current dictionary;
            dict['precision']: array with the precision values;
            dict['recall']: array with the recall values;
            dict['AP']: average precision;
            dict['interpolated precision']: interpolated precision values;
            dict['interpolated recall']: interpolated recall values;
            dict['total positives']: total number of ground truth positives;
            dict['TP']: total number of True Positive detections;
            dict['FP']: total number of False Positive detections;

            if there was no valid ground truth for a specific class (total positives == 0),
            all the associated keys default to None
    """

    # separate bbs per image X class
    _bbs = _group_detections(detected_bbs, groundtruth_bbs)

    # pairwise ious
    _ious = {k: _compute_ious(**v) for k, v in _bbs.items()}

    # accumulate evaluations on a per-class basis
    _evals = defaultdict(lambda: {"scores": [], "matched": [], "NP": []})

    for img_id, class_id in _bbs:
        ev = _evaluate_image(
            _bbs[img_id, class_id]["dt"],
            _bbs[img_id, class_id]["gt"],
            _ious[img_id, class_id],
            iou_threshold,
            max_dets,
            area_range,
        )
        acc = _evals[class_id]
        acc["scores"].append(ev["scores"])
        acc["matched"].append(ev["matched"])
        acc["NP"].append(ev["NP"])

    # now reduce accumulations
    for class_id in _evals:
        acc = _evals[class_id]
        acc["scores"] = np.concatenate(acc["scores"])
        acc["matched"] = np.concatenate(acc["matched"]).astype(bool)
        acc["NP"] = np.sum(acc["NP"])

    res = {}
    # run ap calculation per-class
    for class_id in _evals:
        ev = _evals[class_id]
        res[class_id] = {
            "class": class_id,
            **_compute_ap_recall(ev["scores"], ev["matched"], ev["NP"])
        }
    return res


def _group_detections(dt, gt):
    """ simply group gts and dts on a imageXclass basis """
    bb_info = defaultdict(lambda: {"dt": [], "gt": []})
    for d in dt:
        i_id = d.get_image_name()
        c_id = d.get_class_id()
        bb_info[i_id, c_id]["dt"].append(d)
    for g in gt:
        i_id = g.get_image_name()
        c_id = g.get_class_id()
        bb_info[i_id, c_id]["gt"].append(g)
    return bb_info


def _get_area(a):
    """ COCO does not consider the outer edge as included in the bbox """
    x, y, x2, y2 = a.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
    return (x2 - x) * (y2 - y)


def _jaccard(a, b):
    xa, ya, x2a, y2a = a.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
    xb, yb, x2b, y2b = b.get_absolute_bounding_box(format=BBFormat.XYX2Y2)

    # innermost left x
    xi = max(xa, xb)
    # innermost right x
    x2i = min(x2a, x2b)
    # same for y
    yi = max(ya, yb)
    y2i = min(y2a, y2b)

    # calculate areas
    Aa = max(x2a - xa, 0) * max(y2a - ya, 0)
    Ab = max(x2b - xb, 0) * max(y2b - yb, 0)
    Ai = max(x2i - xi, 0) * max(y2i - yi, 0)
    return Ai / (Aa + Ab - Ai)


def _compute_ious(dt, gt):
    """ compute pairwise ious """

    ious = np.zeros((len(dt), len(gt)))
    for g_idx, g in enumerate(gt):
        for d_idx, d in enumerate(dt):
            ious[d_idx, g_idx] = _jaccard(d, g)
    return ious


def _evaluate_image(dt, gt, ious, iou_threshold, max_dets=None, area_range=None):
    """ use COCO's method to associate detections to ground truths """
    # sort dts by increasing confidence
    dt_sort = np.argsort([-d.get_confidence() for d in dt], kind="stable")

    # sort list of dts and chop by max dets
    dt = [dt[idx] for idx in dt_sort[:max_dets]]
    ious = ious[dt_sort[:max_dets]]

    # generate ignored gt list by area_range
    def _is_ignore(bb):
        if area_range is None:
            return False
        return not (area_range[0] <= _get_area(bb) <= area_range[1])

    gt_ignore = [_is_ignore(g) for g in gt]

    # sort gts by ignore last
    gt_sort = np.argsort(gt_ignore, kind="stable")
    gt = [gt[idx] for idx in gt_sort]
    gt_ignore = [gt_ignore[idx] for idx in gt_sort]
    ious = ious[:, gt_sort]

    gtm = {}
    dtm = {}

    for d_idx, d in enumerate(dt):
        # information about best match so far (m=-1 -> unmatched)
        iou = min(iou_threshold, 1 - 1e-10)
        m = -1
        for g_idx, g in enumerate(gt):
            # if this gt already matched, and not a crowd, continue
            if g_idx in gtm:
                continue
            # if dt matched to reg gt, and on ignore gt, stop
            if m > -1 and gt_ignore[m] == False and gt_ignore[g_idx] == True:
                break
            # continue to next gt unless better match made
            if ious[d_idx, g_idx] < iou:
                continue
            # if match successful and best so far, store appropriately
            iou = ious[d_idx, g_idx]
            m = g_idx
        # if match made store id of match for both dt and gt
        if m == -1:
            continue
        dtm[d_idx] = m
        gtm[m] = d_idx

    # generate ignore list for dts
    dt_ignore = [
        gt_ignore[dtm[d_idx]] if d_idx in dtm else _is_ignore(d) for d_idx, d in enumerate(dt)
    ]

    # get score for non-ignored dts
    scores = [dt[d_idx].get_confidence() for d_idx in range(len(dt)) if not dt_ignore[d_idx]]
    matched = [d_idx in dtm for d_idx in range(len(dt)) if not dt_ignore[d_idx]]

    n_gts = len([g_idx for g_idx in range(len(gt)) if not gt_ignore[g_idx]])
    return {"scores": scores, "matched": matched, "NP": n_gts}


def _compute_ap_recall(scores, matched, NP, recall_thresholds=None):
    """ This curve tracing method has some quirks that do not appear when only unique confidence thresholds
    are used (i.e. Scikit-learn's implementation), however, in order to be consistent, the COCO's method is reproduced. """
    if NP == 0:
        return {
            "precision": None,
            "recall": None,
            "AP": None,
            "interpolated precision": None,
            "interpolated recall": None,
            "total positives": None,
            "TP": None,
            "FP": None
        }

    # by default evaluate on 101 recall levels
    if recall_thresholds is None:
        recall_thresholds = np.linspace(0.0,
                                        1.00,
                                        int(np.round((1.00 - 0.0) / 0.01)) + 1,
                                        endpoint=True)

    # sort in descending score order
    inds = np.argsort(-scores, kind="stable")

    scores = scores[inds]
    matched = matched[inds]

    tp = np.cumsum(matched)
    fp = np.cumsum(~matched)

    rc = tp / NP
    pr = tp / (tp + fp)

    # make precision monotonically decreasing
    i_pr = np.maximum.accumulate(pr[::-1])[::-1]

    rec_idx = np.searchsorted(rc, recall_thresholds, side="left")
    n_recalls = len(recall_thresholds)

    # get interpolated precision values at the evaluation thresholds
    i_pr = np.array([i_pr[r] if r < len(i_pr) else 0 for r in rec_idx])

    return {
        "precision": pr,
        "recall": rc,
        "AP": np.mean(i_pr),
        "interpolated precision": i_pr,
        "interpolated recall": recall_thresholds,
        "total positives": NP,
        "TP": tp[-1] if len(tp) != 0 else 0,
        "FP": fp[-1] if len(fp) != 0 else 0
    }


def get_scale_distribution(bounding_boxes: List) -> Dict[str, Any]:
    """
    Get detailed scale distribution statistics for bounding boxes.
    
    Args:
        bounding_boxes: List of BoundingBox objects.
    
    Returns:
        Dictionary containing:
        - total_count: Total number of boxes
        - by_scale: Per-scale statistics (count, percentage, mean/min/max/std area)
        - summary_text: Human-readable summary string
    """
    if not bounding_boxes:
        return {
            'total_count': 0,
            'by_scale': {},
            'summary_text': 'No bounding boxes provided.'
        }
    
    stats = compute_scale_statistics(bounding_boxes)
    summary = stats.get_summary()
    summary['summary_text'] = str(stats)
    
    return summary


def get_coco_summary_with_scale_details(
    groundtruth_bbs: List,
    detected_bbs: List
) -> Dict[str, Any]:
    """
    Calculate COCO metrics with additional detailed scale-based analysis.
    
    This extends get_coco_summary() with:
    - Detailed GT and detection scale distributions
    - Per-scale TP/FP/FN counts
    - Scale-specific precision and recall
    
    Args:
        groundtruth_bbs: List of ground truth BoundingBox objects.
        detected_bbs: List of detected BoundingBox objects.
    
    Returns:
        Dictionary containing standard COCO metrics plus:
        - gt_scale_distribution: Ground truth scale statistics
        - det_scale_distribution: Detection scale statistics
        - scale_metrics: Detailed metrics per scale category
    """
    # Get standard COCO summary
    base_summary = get_coco_summary(groundtruth_bbs, detected_bbs)
    
    # Add scale distribution analysis
    gt_dist = get_scale_distribution(groundtruth_bbs)
    det_dist = get_scale_distribution(detected_bbs)
    
    # Compute detailed scale-specific metrics
    scale_metrics = _compute_detailed_scale_metrics(groundtruth_bbs, detected_bbs)
    
    return {
        **base_summary,
        'gt_scale_distribution': gt_dist,
        'det_scale_distribution': det_dist,
        'scale_metrics': scale_metrics,
    }


def _compute_detailed_scale_metrics(
    groundtruth_bbs: List,
    detected_bbs: List,
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Compute detailed metrics broken down by scale category.
    
    Args:
        groundtruth_bbs: List of ground truth BoundingBox objects.
        detected_bbs: List of detected BoundingBox objects.
        iou_threshold: IoU threshold for matching (default 0.5).
    
    Returns:
        Dictionary with metrics per scale:
        - small/medium/large each containing:
            - gt_count: Number of ground truth boxes
            - det_count: Number of detection boxes
            - precision: Precision for this scale
            - recall: Recall for this scale
            - f1_score: F1 score for this scale
    """
    scale_configs = {
        'small': (0, COCO_SCALE_THRESHOLDS['small_upper']),
        'medium': (COCO_SCALE_THRESHOLDS['small_upper'], COCO_SCALE_THRESHOLDS['medium_upper']),
        'large': (COCO_SCALE_THRESHOLDS['medium_upper'], np.inf),
    }
    
    result = {}
    
    for scale_name, area_range in scale_configs.items():
        metrics = get_coco_metrics(
            groundtruth_bbs,
            detected_bbs,
            iou_threshold=iou_threshold,
            area_range=area_range,
            max_dets=100
        )
        
        # Count GT and detections in this scale range
        gt_in_scale = [bb for bb in groundtruth_bbs if _is_in_area_range(bb, area_range)]
        det_in_scale = [bb for bb in detected_bbs if _is_in_area_range(bb, area_range)]
        
        # Aggregate metrics across classes
        total_tp = 0
        total_fp = 0
        total_positives = 0
        aps = []
        
        for class_id, class_metrics in metrics.items():
            if class_metrics.get('TP') is not None:
                total_tp += class_metrics['TP']
            if class_metrics.get('FP') is not None:
                total_fp += class_metrics['FP']
            if class_metrics.get('total positives') is not None:
                total_positives += class_metrics['total positives']
            if class_metrics.get('AP') is not None:
                aps.append(class_metrics['AP'])
        
        # Calculate precision, recall, F1
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / total_positives if total_positives > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_ap = np.mean(aps) if aps else np.nan
        
        result[scale_name] = {
            'gt_count': len(gt_in_scale),
            'det_count': len(det_in_scale),
            'total_positives': total_positives,
            'TP': total_tp,
            'FP': total_fp,
            'FN': total_positives - total_tp,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'AP': mean_ap,
            'area_range': area_range,
            'label': get_scale_label(ObjectScale(scale_name)),
        }
    
    return result


def _is_in_area_range(bb, area_range: Tuple[float, float]) -> bool:
    """Check if a bounding box's area falls within the given range."""
    try:
        area = _get_area(bb)
        return area_range[0] <= area <= area_range[1]
    except Exception:
        return False


def get_coco_metrics_by_scale(
    groundtruth_bbs: List,
    detected_bbs: List,
    scale: ObjectScale,
    iou_threshold: float = 0.5,
    max_dets: int = 100
) -> Dict:
    """
    Get COCO metrics for a specific scale category.
    
    Args:
        groundtruth_bbs: List of ground truth BoundingBox objects.
        detected_bbs: List of detected BoundingBox objects.
        scale: ObjectScale category to evaluate.
        iou_threshold: IoU threshold for matching.
        max_dets: Maximum detections per image.
    
    Returns:
        Dictionary with per-class metrics for the specified scale.
    """
    area_range = get_area_range_for_scale(scale)
    return get_coco_metrics(
        groundtruth_bbs,
        detected_bbs,
        iou_threshold=iou_threshold,
        area_range=area_range,
        max_dets=max_dets
    )


def format_scale_metrics_report(
    groundtruth_bbs: List,
    detected_bbs: List
) -> str:
    """
    Generate a formatted text report of scale-based metrics.
    
    Args:
        groundtruth_bbs: List of ground truth BoundingBox objects.
        detected_bbs: List of detected BoundingBox objects.
    
    Returns:
        Formatted string report with scale distribution and metrics.
    """
    summary = get_coco_summary_with_scale_details(groundtruth_bbs, detected_bbs)
    
    lines = []
    lines.append("=" * 70)
    lines.append("COCO EVALUATION REPORT WITH SCALE ANALYSIS")
    lines.append("=" * 70)
    lines.append("")
    
    # Standard metrics
    lines.append("STANDARD COCO METRICS:")
    lines.append("-" * 40)
    lines.append(f"  AP (IoU=0.50:0.95):     {summary['AP']:.4f}")
    lines.append(f"  AP50 (IoU=0.50):        {summary['AP50']:.4f}")
    lines.append(f"  AP75 (IoU=0.75):        {summary['AP75']:.4f}")
    lines.append(f"  APsmall:                {summary['APsmall']:.4f}" if not np.isnan(summary['APsmall']) else "  APsmall:                N/A")
    lines.append(f"  APmedium:               {summary['APmedium']:.4f}" if not np.isnan(summary['APmedium']) else "  APmedium:               N/A")
    lines.append(f"  APlarge:                {summary['APlarge']:.4f}" if not np.isnan(summary['APlarge']) else "  APlarge:                N/A")
    lines.append("")
    lines.append(f"  AR1:                    {summary['AR1']:.4f}")
    lines.append(f"  AR10:                   {summary['AR10']:.4f}")
    lines.append(f"  AR100:                  {summary['AR100']:.4f}")
    lines.append(f"  ARsmall:                {summary['ARsmall']:.4f}" if not np.isnan(summary['ARsmall']) else "  ARsmall:                N/A")
    lines.append(f"  ARmedium:               {summary['ARmedium']:.4f}" if not np.isnan(summary['ARmedium']) else "  ARmedium:               N/A")
    lines.append(f"  ARlarge:                {summary['ARlarge']:.4f}" if not np.isnan(summary['ARlarge']) else "  ARlarge:                N/A")
    lines.append("")
    
    # Ground truth distribution
    lines.append("GROUND TRUTH SCALE DISTRIBUTION:")
    lines.append("-" * 40)
    gt_dist = summary['gt_scale_distribution']
    lines.append(f"  Total GT boxes: {gt_dist['total_count']}")
    for scale_name in ['small', 'medium', 'large']:
        if scale_name in gt_dist.get('by_scale', {}):
            scale_data = gt_dist['by_scale'][scale_name]
            count = scale_data['count']
            pct = scale_data['percentage']
            mean_area = scale_data.get('mean_area')
            mean_str = f", mean area: {mean_area:.1f}" if mean_area else ""
            lines.append(f"  {scale_data['label']:25s}: {count:5d} ({pct:5.1f}%){mean_str}")
    lines.append("")
    
    # Detection distribution
    lines.append("DETECTION SCALE DISTRIBUTION:")
    lines.append("-" * 40)
    det_dist = summary['det_scale_distribution']
    lines.append(f"  Total detections: {det_dist['total_count']}")
    for scale_name in ['small', 'medium', 'large']:
        if scale_name in det_dist.get('by_scale', {}):
            scale_data = det_dist['by_scale'][scale_name]
            count = scale_data['count']
            pct = scale_data['percentage']
            mean_area = scale_data.get('mean_area')
            mean_str = f", mean area: {mean_area:.1f}" if mean_area else ""
            lines.append(f"  {scale_data['label']:25s}: {count:5d} ({pct:5.1f}%){mean_str}")
    lines.append("")
    
    # Detailed scale metrics
    lines.append("DETAILED SCALE METRICS (IoU=0.50):")
    lines.append("-" * 40)
    scale_metrics = summary.get('scale_metrics', {})
    
    header = f"{'Scale':15s} {'GT':>6s} {'Det':>6s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s}"
    lines.append(header)
    lines.append("-" * len(header))
    
    for scale_name in ['small', 'medium', 'large']:
        if scale_name in scale_metrics:
            m = scale_metrics[scale_name]
            lines.append(
                f"{scale_name.capitalize():15s} "
                f"{m['gt_count']:6d} "
                f"{m['det_count']:6d} "
                f"{m['TP']:6d} "
                f"{m['FP']:6d} "
                f"{m['FN']:6d} "
                f"{m['precision']:7.3f} "
                f"{m['recall']:7.3f} "
                f"{m['f1_score']:7.3f}"
            )
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)
