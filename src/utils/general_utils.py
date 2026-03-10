import fnmatch
import os
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtCore, QtGui
from src.utils.enumerators import BBFormat
from src.utils.object_scale import (
    ObjectScale,
    classify_scale,
    get_scale_color_bgr,
    get_scale_color_rgb,
    get_scale_color_normalized,
    get_scale_label,
    ScaleStatistics,
    SCALE_COLORS_BGR,
    SCALE_COLORS_NORMALIZED,
)


def get_classes_from_txt_file(filepath_classes_det):
    classes = {}
    f = open(filepath_classes_det, 'r')
    id_class = 0
    for id_class, line in enumerate(f.readlines()):
        classes[id_class] = line.replace('\n', '')
    f.close()
    return classes


def replace_id_with_classes(bounding_boxes, filepath_classes_det):
    classes = get_classes_from_txt_file(filepath_classes_det)
    for bb in bounding_boxes:
        if not is_str_int(bb.get_class_id()):
            print(
                f'Warning: Class id represented in the {filepath_classes_det} is not a valid integer.'
            )
            return bounding_boxes
        class_id = int(bb.get_class_id())
        if class_id not in range(len(classes)):
            print(
                f'Warning: Class id {class_id} is not in the range of classes specified in the file {filepath_classes_det}.'
            )
            return bounding_boxes
        bb._class_id = classes[class_id]
    return bounding_boxes


def convert_box_xywh2xyxy(box):
    arr = box.copy()
    arr[:, 2] += arr[:, 0]
    arr[:, 3] += arr[:, 1]
    return arr


def convert_box_xyxy2xywh(box):
    arr = box.copy()
    arr[:, 2] -= arr[:, 0]
    arr[:, 3] -= arr[:, 1]
    return arr


# size => (width, height) of the image
# box => (X1, X2, Y1, Y2) of the bounding box
def convert_to_relative_values(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    cx = (box[1] + box[0]) / 2.0
    cy = (box[3] + box[2]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = cx * dw
    y = cy * dh
    w = w * dw
    h = h * dh
    # YOLO's format
    # x,y => (bounding_box_center)/width_of_the_image
    # w => bounding_box_width / width_of_the_image
    # h => bounding_box_height / height_of_the_image
    return (x, y, w, h)


# size => (width, height) of the image
# box => (centerX, centerY, w, h) of the bounding box relative to the image
def convert_to_absolute_values(size, box):
    w_box = size[0] * box[2]
    h_box = size[1] * box[3]

    x1 = (float(box[0]) * float(size[0])) - (w_box / 2)
    y1 = (float(box[1]) * float(size[1])) - (h_box / 2)
    x2 = x1 + w_box
    y2 = y1 + h_box
    return (round(x1), round(y1), round(x2), round(y2))


def add_bb_into_image(image, bb, color=(255, 0, 0), thickness=2, label=None):
    r = int(color[0])
    g = int(color[1])
    b = int(color[2])

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    fontThickness = 1

    x1, y1, x2, y2 = bb.get_absolute_bounding_box(BBFormat.XYX2Y2)
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    cv2.rectangle(image, (x1, y1), (x2, y2), (b, g, r), thickness)
    # Add label
    if label is not None:
        # Get size of the text box
        (tw, th) = cv2.getTextSize(label, font, fontScale, fontThickness)[0]
        # Top-left coord of the textbox
        (xin_bb, yin_bb) = (x1 + thickness, y1 - th + int(12.5 * fontScale))
        # Checking position of the text top-left (outside or inside the bb)
        if yin_bb - th <= 0:  # if outside the image
            yin_bb = y1 + th  # put it inside the bb
        r_Xin = x1 - int(thickness / 2)
        r_Yin = y1 - th - int(thickness / 2)
        # Draw filled rectangle to put the text in it
        cv2.rectangle(image, (r_Xin, r_Yin - thickness),
                      (r_Xin + tw + thickness * 3, r_Yin + th + int(12.5 * fontScale)), (b, g, r),
                      -1)
        cv2.putText(image, label, (xin_bb, yin_bb), font, fontScale, (0, 0, 0), fontThickness,
                    cv2.LINE_AA)
    return image


def add_bb_into_image_with_scale_color(
    image,
    bb,
    thickness: int = 2,
    label: Optional[str] = None,
    show_scale_in_label: bool = True
):
    """
    Add a bounding box to an image with color based on its COCO scale category.
    
    Colors:
    - Small objects (≤32²px): Blue
    - Medium objects (32²-96²px): Green
    - Large objects (>96²px): Red
    
    Args:
        image: OpenCV image (numpy array).
        bb: BoundingBox object.
        thickness: Line thickness for the bounding box.
        label: Optional label text. If None, uses class_id.
        show_scale_in_label: If True, appends scale category to label.
    
    Returns:
        Image with bounding box drawn.
    """
    # Get scale-based color (BGR for OpenCV)
    try:
        scale = bb.get_scale()
        color_bgr = get_scale_color_bgr(scale)
    except Exception:
        # Fallback: calculate area manually
        try:
            x1, y1, x2, y2 = bb.get_absolute_bounding_box(BBFormat.XYX2Y2)
            area = (x2 - x1) * (y2 - y1)
            scale = classify_scale(area)
            color_bgr = get_scale_color_bgr(scale)
        except Exception:
            scale = ObjectScale.UNKNOWN
            color_bgr = SCALE_COLORS_BGR[ObjectScale.UNKNOWN]
    
    # Build label
    if label is None:
        try:
            label = str(bb.get_class_id())
        except Exception:
            label = ""
    
    if show_scale_in_label and label:
        label = f"{label} [{scale.value}]"
    elif show_scale_in_label:
        label = f"[{scale.value}]"
    
    # Convert BGR to RGB for the existing function
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    
    return add_bb_into_image(image, bb, color=color_rgb, thickness=thickness, label=label)


def draw_bbs_with_scale_colors(
    image,
    bounding_boxes: List,
    thickness: int = 2,
    show_labels: bool = True,
    show_scale_in_label: bool = True
):
    """
    Draw multiple bounding boxes on an image with scale-based colors.
    
    Args:
        image: OpenCV image (numpy array) or path to image file.
        bounding_boxes: List of BoundingBox objects.
        thickness: Line thickness.
        show_labels: If True, show class labels.
        show_scale_in_label: If True, append scale category to labels.
    
    Returns:
        Image with all bounding boxes drawn.
    """
    if isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            raise ValueError(f"Could not load image: {image}")
    
    # Make a copy to avoid modifying the original
    result = image.copy()
    
    if not bounding_boxes:
        return result
    
    for bb in bounding_boxes:
        if bb is None:
            continue
        
        label = None
        if show_labels:
            try:
                label = str(bb.get_class_id())
            except Exception:
                label = None
        
        try:
            result = add_bb_into_image_with_scale_color(
                result,
                bb,
                thickness=thickness,
                label=label,
                show_scale_in_label=show_scale_in_label
            )
        except Exception:
            # Skip boxes that can't be drawn
            continue
    
    return result


def create_scale_legend_image(
    width: int = 300,
    height: int = 120,
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    Create an image showing the scale color legend.
    
    Args:
        width: Legend image width.
        height: Legend image height.
        background_color: Background color (BGR).
    
    Returns:
        OpenCV image (numpy array) with the legend.
    """
    img = np.full((height, width, 3), background_color, dtype=np.uint8)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    
    y_start = 25
    y_step = 30
    box_size = 20
    x_box = 10
    x_text = 40
    
    scales = [ObjectScale.SMALL, ObjectScale.MEDIUM, ObjectScale.LARGE]
    
    for i, scale in enumerate(scales):
        y = y_start + i * y_step
        color = SCALE_COLORS_BGR[scale]
        
        # Draw color box
        cv2.rectangle(img, (x_box, y - box_size // 2), (x_box + box_size, y + box_size // 2), color, -1)
        cv2.rectangle(img, (x_box, y - box_size // 2), (x_box + box_size, y + box_size // 2), (0, 0, 0), 1)
        
        # Draw label
        label = get_scale_label(scale)
        cv2.putText(img, label, (x_text, y + 5), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
    
    return img


def add_scale_legend_to_image(
    image,
    position: str = 'top-right',
    margin: int = 10
) -> np.ndarray:
    """
    Add a scale color legend overlay to an image.
    
    Args:
        image: OpenCV image (numpy array).
        position: Legend position ('top-right', 'top-left', 'bottom-right', 'bottom-left').
        margin: Margin from image edge.
    
    Returns:
        Image with legend overlay.
    """
    result = image.copy()
    legend = create_scale_legend_image()
    
    lh, lw = legend.shape[:2]
    ih, iw = result.shape[:2]
    
    # Calculate position
    pos_lower = (position or 'top-right').lower().strip()
    
    if 'left' in pos_lower:
        x = margin
    else:  # right
        x = iw - lw - margin
    
    if 'bottom' in pos_lower:
        y = ih - lh - margin
    else:  # top
        y = margin
    
    # Ensure we don't go out of bounds
    x = max(0, min(x, iw - lw))
    y = max(0, min(y, ih - lh))
    
    # Blend legend onto image
    result[y:y + lh, x:x + lw] = cv2.addWeighted(result[y:y + lh, x:x + lw], 0.3, legend, 0.7, 0)
    
    return result


def remove_file_extension(filename):
    return os.path.join(os.path.dirname(filename), os.path.splitext(filename)[0])


def get_files_dir(directory, extensions=['*']):
    ret = []
    for extension in extensions:
        if extension == '*':
            ret += [f for f in os.listdir(directory)]
            continue
        elif extension is None:
            # accepts all extensions
            extension = ''
        elif '.' not in extension:
            extension = f'.{extension}'
        ret += [f for f in os.listdir(directory) if f.lower().endswith(extension.lower())]
    return ret


def remove_file_extension(filename):
    return os.path.join(os.path.dirname(filename), os.path.splitext(filename)[0])


def image_to_pixmap(image):
    image = image.astype(np.uint8)
    if image.shape[2] == 4:
        qformat = QtGui.QImage.Format_RGBA8888
    else:
        qformat = QtGui.QImage.Format_RGB888

    image = QtGui.QImage(image.data, image.shape[1], image.shape[0], image.strides[0], qformat)
    # image= image.rgbSwapped()
    return QtGui.QPixmap(image)


def show_image_in_qt_component(image, label_component):
    pix = image_to_pixmap((image).astype(np.uint8))
    label_component.setPixmap(pix)
    label_component.setAlignment(QtCore.Qt.AlignCenter)


def get_files_recursively(directory, extension="*"):
    files = [
        os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(directory)
        for f in get_files_dir(directory, [extension])
    ]
    # Disconsider hidden files, such as .DS_Store in the MAC OS
    ret = [f for f in files if not os.path.basename(f).startswith('.')]
    return ret


def is_str_int(s):
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()


def get_file_name_only(file_path):
    if file_path is None:
        return ''
    return os.path.splitext(os.path.basename(file_path))[0]


# allowed_extensions is used only when match_extension=False
def find_file(directory, file_name, match_extension=True, allowed_extensions=[]):
    if os.path.isdir(directory) is False:
        return None
    for dirpath, dirnames, files in os.walk(directory):
        for f in files:
            f1 = os.path.basename(f)
            f2 = file_name
            if match_extension:
                match = f1 == f2
            else:
                f1 = os.path.splitext(f1)[0]
                f2 = os.path.splitext(f2)[0]
                f_ext = os.path.splitext(f)[-1].lower()
                match = f1 == f2 and (len(allowed_extensions) == 0 or f_ext in allowed_extensions)
            if match:
                return os.path.join(dirpath, os.path.basename(f))
    return None


def find_image_file(directory, file_name):
    return find_file(directory, file_name, False, [".bmp", ".jpg", ".jpeg", ".png"])


def get_image_resolution(image_file):
    if image_file is None or not os.path.isfile(image_file):
        print(f'Warning: Path {image_file} not found.')
        return None
    img = cv2.imread(image_file)
    if img is None:
        print(f'Warning: Error loading the image {image_file}.')
        return None
    h, w, _ = img.shape
    return {'height': h, 'width': w}


def draw_bb_into_image(image, boundingBox, color, thickness, label=None):
    if isinstance(image, str):
        image = cv2.imread(image)

    r = int(color[0])
    g = int(color[1])
    b = int(color[2])

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    fontThickness = 1

    xIn = boundingBox[0]
    yIn = boundingBox[1]
    cv2.rectangle(image, (boundingBox[0], boundingBox[1]), (boundingBox[2], boundingBox[3]),
                  (b, g, r), thickness)
    # Add label
    if label is not None:
        # Get size of the text box
        (tw, th) = cv2.getTextSize(label, font, fontScale, fontThickness)[0]
        # Top-left coord of the textbox
        (xin_bb, yin_bb) = (xIn + thickness, yIn - th + int(12.5 * fontScale))
        # Checking position of the text top-left (outside or inside the bb)
        if yin_bb - th <= 0:  # if outside the image
            yin_bb = yIn + th  # put it inside the bb
        r_Xin = xIn - int(thickness / 2)
        r_Yin = yin_bb - th - int(thickness / 2)
        # Draw filled rectangle to put the text in it
        cv2.rectangle(image, (r_Xin, r_Yin - thickness),
                      (r_Xin + tw + thickness * 3, r_Yin + th + int(12.5 * fontScale)), (b, g, r),
                      -1)
        cv2.putText(image, label, (xin_bb, yin_bb), font, fontScale, (0, 0, 0), fontThickness,
                    cv2.LINE_AA)
    return image


def plot_bb_per_classes(dict_bbs_per_class,
                        horizontally=True,
                        rotation=0,
                        show=False,
                        extra_title=''):
    plt.close()
    if horizontally:
        ypos = np.arange(len(dict_bbs_per_class.keys()))
        plt.barh(ypos, dict_bbs_per_class.values())
        plt.yticks(ypos, dict_bbs_per_class.keys())
        plt.xlabel('amount of bounding boxes')
        plt.ylabel('classes')
    else:
        plt.bar(dict_bbs_per_class.keys(), dict_bbs_per_class.values())
        plt.xlabel('classes')
        plt.ylabel('amount of bounding boxes')
    plt.xticks(rotation=rotation)
    title = f'Distribution of bounding boxes per class {extra_title}'
    plt.title(title)
    if show:
        fig = plt.gcf()
        fig.tight_layout()
        # Use manager.set_window_title for matplotlib >= 3.4 compatibility
        if hasattr(fig.canvas, 'manager') and fig.canvas.manager is not None:
            fig.canvas.manager.set_window_title(title)
        plt.show()
    return plt


def plot_bb_per_scale(
    bounding_boxes: List,
    show: bool = False,
    save_path: Optional[str] = None,
    extra_title: str = '',
    use_scale_colors: bool = True
):
    """
    Plot bounding box distribution by COCO scale category.
    
    Args:
        bounding_boxes: List of BoundingBox objects.
        show: If True, display the plot.
        save_path: If provided, save plot to this path.
        extra_title: Additional text to append to title.
        use_scale_colors: If True, use scale-specific colors for bars.
    
    Returns:
        matplotlib pyplot object.
    """
    plt.close()
    
    # Compute statistics
    stats = ScaleStatistics()
    for bb in bounding_boxes or []:
        if bb is None:
            continue
        try:
            area = bb.get_area()
            stats.add_box(area)
        except Exception:
            stats.add_box(None)
    
    # Prepare data
    scales = [ObjectScale.SMALL, ObjectScale.MEDIUM, ObjectScale.LARGE]
    labels = [get_scale_label(s) for s in scales]
    counts = [stats.get_count(s) for s in scales]
    percentages = [stats.get_percentage(s) for s in scales]
    
    if use_scale_colors:
        colors = [SCALE_COLORS_NORMALIZED[s] for s in scales]
    else:
        colors = None
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart of counts
    bars = ax1.bar(labels, counts, color=colors, edgecolor='black')
    ax1.set_xlabel('Scale Category')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Bounding Box Count by Scale {extra_title}')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.annotate(f'{count}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')
    
    # Pie chart of distribution
    if sum(counts) > 0:
        ax2.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title(f'Scale Distribution {extra_title}')
    else:
        ax2.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(f'Scale Distribution {extra_title}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return plt


def plot_scale_comparison(
    gt_bbs: List,
    det_bbs: List,
    show: bool = False,
    save_path: Optional[str] = None,
    title: str = 'Scale Distribution: Ground Truth vs Detections'
):
    """
    Plot comparison of scale distributions between ground truth and detections.
    
    Args:
        gt_bbs: List of ground truth BoundingBox objects.
        det_bbs: List of detected BoundingBox objects.
        show: If True, display the plot.
        save_path: If provided, save plot to this path.
        title: Plot title.
    
    Returns:
        matplotlib pyplot object.
    """
    plt.close()
    
    # Compute statistics for both
    gt_stats = ScaleStatistics()
    det_stats = ScaleStatistics()
    
    for bb in gt_bbs or []:
        if bb is None:
            continue
        try:
            gt_stats.add_box(bb.get_area())
        except Exception:
            gt_stats.add_box(None)
    
    for bb in det_bbs or []:
        if bb is None:
            continue
        try:
            det_stats.add_box(bb.get_area())
        except Exception:
            det_stats.add_box(None)
    
    scales = [ObjectScale.SMALL, ObjectScale.MEDIUM, ObjectScale.LARGE]
    labels = ['Small', 'Medium', 'Large']
    
    gt_counts = [gt_stats.get_count(s) for s in scales]
    det_counts = [det_stats.get_count(s) for s in scales]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width / 2, gt_counts, width, label='Ground Truth', color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width / 2, det_counts, width, label='Detections', color='coral', edgecolor='black')
    
    ax.set_xlabel('Scale Category')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return plt


def plot_scale_metrics(
    scale_metrics: Dict,
    show: bool = False,
    save_path: Optional[str] = None,
    title: str = 'Detection Performance by Scale'
):
    """
    Plot precision, recall, and F1 score by scale category.
    
    Args:
        scale_metrics: Dictionary from get_coco_summary_with_scale_details()['scale_metrics'].
        show: If True, display the plot.
        save_path: If provided, save plot to this path.
        title: Plot title.
    
    Returns:
        matplotlib pyplot object.
    """
    plt.close()
    
    if not scale_metrics:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No metrics data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return plt
    
    scales = ['small', 'medium', 'large']
    labels = ['Small', 'Medium', 'Large']
    
    precision = []
    recall = []
    f1 = []
    
    for scale in scales:
        if scale in scale_metrics:
            precision.append(scale_metrics[scale].get('precision', 0))
            recall.append(scale_metrics[scale].get('recall', 0))
            f1.append(scale_metrics[scale].get('f1_score', 0))
        else:
            precision.append(0)
            recall.append(0)
            f1.append(0)
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='steelblue', edgecolor='black')
    bars2 = ax.bar(x, recall, width, label='Recall', color='coral', edgecolor='black')
    bars3 = ax.bar(x + width, f1, width, label='F1 Score', color='seagreen', edgecolor='black')
    
    ax.set_xlabel('Scale Category')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return plt
