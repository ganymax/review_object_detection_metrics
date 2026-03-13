"""
Details dialog for displaying bounding box statistics and image visualization.

This module provides the Details_Dialog class which displays:
- Bounding box statistics (count, average area, distribution by class)
- Image viewer with bounding boxes drawn using COCO scale-based colors:
  - Small objects (≤32²px): Blue
  - Medium objects (32²-96²px): Green  
  - Large objects (>96²px): Red
- Scale distribution statistics and plotting
- Bias profiling with dominant color markers for medium/large boxes
"""

import os
import random

import cv2
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QFileDialog, QMainWindow
from src.bounding_box import BoundingBox
from src.ui.details_ui import Ui_Dialog as Details_UI
from src.utils import general_utils
from src.utils.enumerators import BBType
from src.utils.general_utils import (
    add_bb_into_image,
    add_bb_into_image_with_scale_color,
    get_files_dir,
    remove_file_extension,
    show_image_in_qt_component,
    plot_bb_per_scale,
)
from src.utils.object_scale import ObjectScale, get_scale_label


class Details_Dialog(QMainWindow, Details_UI):
    """
    Dialog window for displaying bounding box statistics and visualizing images.
    
    Features:
    - Display statistics: total boxes, images, average area, class distribution
    - Display scale distribution using COCO standard (small/medium/large)
    - Image viewer with bounding boxes colored by scale category
    - Plot generation for class and scale distributions
    
    Scale Color Coding (COCO Standard):
    - Small (area ≤ 32²px = 1024px): Blue RGB(100, 100, 255)
    - Medium (32²px < area ≤ 96²px = 9216px): Green RGB(100, 255, 100)
    - Large (area > 96²px): Red RGB(255, 100, 100)
    """
    
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        # initialize variables
        self.dir_images = ''
        self.gt_annotations = None
        self.det_annotations = None
        self.text_statistics = '<b>#TYPE_BB#:</b><br>'
        self.text_statistics += '<br>* A total of <b>#TOTAL_BB#</b> bounding boxes were found in <b>#TOTAL_IMAGES#</b> images.'
        self.text_statistics += '<br>* The average area of the bounding boxes is <b>#AVERAGE_AREA_BB#</b> pixels.'
        self.text_statistics += '<br>* The amount of bounding boxes per class is:'
        self.text_statistics += '<br>#AMOUNT_BB_PER_CLASS#'
        self.text_statistics += '<br><br>* <b>Scale distribution (COCO standard):</b>'
        self.text_statistics += '<br>#SCALE_DISTRIBUTION#'
        self.text_statistics += '<br><br><i>Note: Bounding boxes are color-coded by scale:</i>'
        self.text_statistics += '<br><span style="color:blue">■</span> Small (≤32²px) '
        self.text_statistics += '<span style="color:green">■</span> Medium (32²-96²px) '
        self.text_statistics += '<span style="color:red">■</span> Large (>96²px)'
        self.lbl_sample_image.setScaledContents(True)
        # set maximum and minimum size
        self.setMaximumHeight(self.height())
        self.setMaximumWidth(self.width())
        # set selected image based on the list of images
        self.selected_image_index = 0

    def initialize_ui(self):
        """
        Initialize the UI components with statistics and prepare image display.
        
        Computes and displays:
        - Total bounding box count and image count
        - Average bounding box area
        - Distribution by object class
        - Distribution by COCO scale category (small/medium/large)
        """
        # clear all information
        self.txb_statistics.setText('')
        self.lbl_sample_image.setText('')
        self.btn_previous_image.setEnabled(False)
        self.btn_next_image.setEnabled(False)
        # Create text with ground truth statistics
        if self.type_bb == BBType.GROUND_TRUTH:
            stats = self.text_statistics.replace('#TYPE_BB#', 'Ground Truth')
            self.annot_obj = self.gt_annotations
        elif self.type_bb == BBType.DETECTED:
            stats = self.text_statistics.replace('#TYPE_BB#', 'Detections')
            self.annot_obj = self.det_annotations
        self.chb_det_bb.setVisible(False)
        self.chb_gt_bb.setVisible(False)
        if self.det_annotations is not None and self.det_annotations != []:
            self.chb_det_bb.setVisible(True)
        if self.gt_annotations is not None and self.gt_annotations != []:
            self.chb_gt_bb.setVisible(True)
        stats = stats.replace('#TOTAL_BB#', str(len(self.annot_obj)))
        stats = stats.replace('#TOTAL_IMAGES#', str(BoundingBox.get_total_images(self.annot_obj)))
        stats = stats.replace('#AVERAGE_AREA_BB#',
                              '%.2f' % BoundingBox.get_average_area(self.annot_obj))
        # Get amount of bounding boxes per class
        self.bb_per_class = BoundingBox.get_amount_bounding_box_all_classes(self.annot_obj)
        amount_bb_per_class = 'No class found'
        if len(self.bb_per_class) > 0:
            amount_bb_per_class = ''
            longest_class_name = len(max(self.bb_per_class.keys(), key=len))
            for c, amount in self.bb_per_class.items():
                c = c.ljust(longest_class_name, ' ')
                amount_bb_per_class += f'   {c} : {amount}<br>'
        stats = stats.replace('#AMOUNT_BB_PER_CLASS#', amount_bb_per_class)
        
        # Get scale distribution statistics (COCO standard)
        self.scale_stats = BoundingBox.get_scale_statistics(self.annot_obj)
        self.bb_per_scale = BoundingBox.get_amount_bounding_box_by_scale(self.annot_obj)
        scale_distribution = ''
        for scale in [ObjectScale.SMALL, ObjectScale.MEDIUM, ObjectScale.LARGE]:
            count = self.scale_stats.get_count(scale)
            pct = self.scale_stats.get_percentage(scale)
            label = get_scale_label(scale)
            # Color code the scale names in the statistics
            if scale == ObjectScale.SMALL:
                color = 'blue'
            elif scale == ObjectScale.MEDIUM:
                color = 'green'
            else:
                color = 'red'
            scale_distribution += f'   <span style="color:{color}">{label}</span>: {count} ({pct:.1f}%)<br>'
        stats = stats.replace('#SCALE_DISTRIBUTION#', scale_distribution)
        
        self.txb_statistics.setText(stats)

        # get first image file and show it
        if os.path.isdir(self.dir_images):
            self.image_files = get_files_dir(
                self.dir_images, extensions=['jpg', 'jpge', 'png', 'bmp', 'tiff', 'tif'])
            if len(self.image_files) > 0:
                self.selected_image_index = 0
            else:
                self.selected_image_index = -1
        else:
            self.image_files = []
            self.selected_image_index = -1
        self.show_image()

    def show_image(self):
        if self.selected_image_index not in range(len(self.image_files)):
            self.btn_save_image.setEnabled(False)
            self.chb_gt_bb.setEnabled(False)
            self.chb_det_bb.setEnabled(False)
            self.lbl_sample_image.clear()
            self.lbl_image_file_name.setText('no image to show')
            return
        # Get all annotations and detections from this file
        if self.annot_obj is not None:
            # If Ground truth, bb will be drawn in green, red otherwise
            self.btn_previous_image.setEnabled(True)
            self.btn_next_image.setEnabled(True)
            self.btn_save_image.setEnabled(True)
            self.chb_gt_bb.setEnabled(True)
            self.chb_det_bb.setEnabled(True)
            self.lbl_image_file_name.setText(self.image_files[self.selected_image_index])
            # Draw bounding boxes
            self.loaded_image = self.draw_bounding_boxes()
            # Show image
            show_image_in_qt_component(self.loaded_image, self.lbl_sample_image)

    def draw_bounding_boxes(self):
        """
        Draw bounding boxes on the current image using COCO scale-based colors.
        
        Bounding boxes are colored according to their COCO scale category:
        - Small (area ≤ 32²px): Blue RGB(100, 100, 255)
        - Medium (32²px < area ≤ 96²px): Green RGB(100, 255, 100)
        - Large (area > 96²px): Red RGB(255, 100, 100)
        
        For medium and large boxes, a crosshair marker is drawn at the center
        of gravity of the dominant color region when bounding boxes are displayed.
        
        The scale category is determined by the absolute pixel area of each
        bounding box, following the COCO evaluation standard.
        
        Returns:
            numpy.ndarray: Image with bounding boxes drawn in scale-based colors.
        """
        # Load image to obtain a clean image (without BBs)
        img_path = os.path.join(self.dir_images, self.image_files[self.selected_image_index])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Get bounding boxes of the loaded image
        img_name = self.image_files[self.selected_image_index]
        img_name = general_utils.get_file_name_only(img_name)
        
        # Enable color markers when any bounding boxes are being drawn
        drawing_gt = self.chb_gt_bb.isChecked() and self.gt_annotations is not None
        drawing_det = self.chb_det_bb.isChecked() and self.det_annotations is not None
        show_markers = drawing_gt or drawing_det
        
        # Add ground truth bounding boxes with scale-based colors
        if drawing_gt:
            bboxes = BoundingBox.get_bounding_boxes_by_image_name(self.gt_annotations, img_name)
            for bb in bboxes:
                img = add_bb_into_image_with_scale_color(
                    img, bb, thickness=2, label=None, show_scale_in_label=False,
                    show_color_marker=show_markers,
                    marker_size=8, color_tolerance=40
                )
        
        # Add detection bounding boxes with scale-based colors
        if drawing_det:
            bboxes = BoundingBox.get_bounding_boxes_by_image_name(self.det_annotations, img_name)
            for bb in bboxes:
                img = add_bb_into_image_with_scale_color(
                    img, bb, thickness=2, label=None, show_scale_in_label=False,
                    show_color_marker=show_markers,
                    marker_size=8, color_tolerance=40
                )
        
        return img

    def show_dialog(self, type_bb, gt_annotations=None, det_annotations=None, dir_images=None):
        self.type_bb = type_bb
        self.gt_annotations = gt_annotations
        self.det_annotations = det_annotations
        self.dir_images = dir_images
        self.initialize_ui()
        self.show()

    def btn_plot_bb_per_classes_clicked(self):
        """
        Handle click event for plotting bounding box distribution by class.
        
        Displays a bar chart showing the count of bounding boxes per object class.
        """
        general_utils.plot_bb_per_classes(self.bb_per_class,
                                          horizontally=False,
                                          rotation=90,
                                          show=True)

    def btn_plot_bb_per_scale_clicked(self):
        """
        Handle click event for plotting bounding box distribution by COCO scale.
        
        Displays charts showing the distribution of bounding boxes across
        COCO scale categories (small, medium, large) with scale-appropriate colors.
        """
        plot_bb_per_scale(self.annot_obj, show=True)

    # def btn_load_random_image_clicked(self):
    #     self.load_random_image()

    def btn_next_image_clicked(self):
        # If reached the last image, set index to start over
        if self.selected_image_index == len(self.image_files) - 1:
            self.selected_image_index = 0
        else:
            self.selected_image_index += 1
        self.show_image()

    def btn_previous_image_clicked(self):
        if self.selected_image_index == 0:
            self.selected_image_index = len(self.image_files) - 1
        else:
            self.selected_image_index -= 1
        self.show_image()

    def btn_save_image_clicked(self):
        dict_formats = {
            'PNG Image (*.png)': 'png',
            'JPEG Image (*.jpg, *.jpeg)': 'jpg',
            'TIFF Image (*.tif, *.tiff)': 'tif'
        }
        formats = ';;'.join(dict_formats.keys())
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, file_extension = QFileDialog.getSaveFileName(self,
                                                                "Save Image File",
                                                                "",
                                                                formats,
                                                                options=options)
        if file_name != '':
            # the extension was not informed, so add it
            if '.' not in file_name:
                file_name = file_name + '.' + dict_formats[file_extension]
            cv2.imwrite(file_name, cv2.cvtColor(self.loaded_image, cv2.COLOR_RGB2BGR))

    def chb_det_bb_clicked(self, state):
        # Draw bounding boxes
        self.loaded_image = self.draw_bounding_boxes()
        # Show image
        show_image_in_qt_component(self.loaded_image, self.lbl_sample_image)

    def chb_gt_bb_clicked(self, state):
        # Draw bounding boxes
        self.loaded_image = self.draw_bounding_boxes()
        # Show image
        show_image_in_qt_component(self.loaded_image, self.lbl_sample_image)


