"""
Details dialog for displaying bounding box statistics and image viewer.

This module provides a PyQt5 dialog for viewing dataset statistics and 
browsing images with bounding box overlays. Bounding boxes are color-coded
based on their COCO scale category:
- Small (area ≤ 32²px): Blue
- Medium (32² < area ≤ 96²px): Green  
- Large (area > 96²px): Red
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
    add_scale_legend_to_image,
)
from src.utils.object_scale import ObjectScale


class Details_Dialog(QMainWindow, Details_UI):
    """
    Dialog for displaying bounding box statistics and browsing annotated images.
    
    This dialog shows:
    - Statistics about the loaded annotations (count, average area, per-class distribution)
    - An image viewer with bounding box overlays
    - Bounding boxes are color-coded by COCO scale category (small/medium/large)
    
    Attributes:
        dir_images: Directory containing the dataset images.
        gt_annotations: List of ground truth BoundingBox objects.
        det_annotations: List of detection BoundingBox objects.
        type_bb: Current bounding box type being displayed (GT or Detection).
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
        # Add scale distribution to statistics
        self.text_statistics += '<br><br>* Bounding boxes by scale (COCO standard):'
        self.text_statistics += '<br>#AMOUNT_BB_PER_SCALE#'
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
        - Total bounding box count
        - Total image count
        - Average bounding box area
        - Per-class bounding box distribution
        - Per-scale bounding box distribution (COCO standard: small/medium/large)
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
        
        # Get amount of bounding boxes per scale (COCO standard)
        self.bb_per_scale = BoundingBox.get_amount_bounding_box_by_scale(self.annot_obj)
        scale_colors = {
            'small': '<span style="color: blue;">●</span>',
            'medium': '<span style="color: green;">●</span>',
            'large': '<span style="color: red;">●</span>',
        }
        scale_labels = {
            'small': 'Small (≤32²px)',
            'medium': 'Medium (32²-96²px)',
            'large': 'Large (>96²px)',
        }
        amount_bb_per_scale = ''
        for scale_name in ['small', 'medium', 'large']:
            if scale_name in self.bb_per_scale:
                color_dot = scale_colors.get(scale_name, '')
                label = scale_labels.get(scale_name, scale_name)
                amount = self.bb_per_scale[scale_name]
                amount_bb_per_scale += f'   {color_dot} {label}: {amount}<br>'
            else:
                color_dot = scale_colors.get(scale_name, '')
                label = scale_labels.get(scale_name, scale_name)
                amount_bb_per_scale += f'   {color_dot} {label}: 0<br>'
        stats = stats.replace('#AMOUNT_BB_PER_SCALE#', amount_bb_per_scale)
        
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
        Draw bounding boxes on the currently selected image.
        
        Bounding boxes are color-coded based on their COCO scale category:
        - Small objects (area ≤ 32²px): Blue (RGB: 100, 100, 255)
        - Medium objects (32² < area ≤ 96²px): Green (RGB: 100, 255, 100)
        - Large objects (area > 96²px): Red (RGB: 255, 100, 100)
        
        The scale is determined by the bounding box area in absolute pixels,
        following the COCO dataset standard thresholds.
        
        Returns:
            numpy.ndarray: Image with bounding boxes drawn, in RGB format.
        """
        # Load image to obtain a clean image (without BBs)
        img_path = os.path.join(self.dir_images, self.image_files[self.selected_image_index])
        img = cv2.imread(img_path)
        if img is None:
            # Return empty image if file cannot be loaded
            import numpy as np
            return np.zeros((480, 640, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get bounding boxes of the loaded image
        img_name = self.image_files[self.selected_image_index]
        img_name = general_utils.get_file_name_only(img_name)
        
        # Track if any bounding boxes were drawn (for legend display)
        has_bboxes = False
        
        # Add ground truth bounding boxes with scale-based colors
        if self.chb_gt_bb.isChecked() and self.gt_annotations is not None:
            bboxes = BoundingBox.get_bounding_boxes_by_image_name(self.gt_annotations, img_name)
            # Draw bounding boxes with COCO scale-based colors
            for bb in bboxes:
                if bb is None:
                    continue
                # Use scale-based color coding
                # Label shows class name and scale category
                label = f"{bb.get_class_id()} [GT]"
                img = add_bb_into_image_with_scale_color(
                    img, bb, thickness=2, label=label, show_scale_in_label=True
                )
                has_bboxes = True
        
        # Add detection bounding boxes with scale-based colors
        if self.chb_det_bb.isChecked() and self.det_annotations is not None:
            bboxes = BoundingBox.get_bounding_boxes_by_image_name(self.det_annotations, img_name)
            # Draw bounding boxes with COCO scale-based colors
            for bb in bboxes:
                if bb is None:
                    continue
                # Use scale-based color coding
                # Label shows class name, confidence, and scale category
                confidence = bb.get_confidence()
                if confidence is not None:
                    label = f"{bb.get_class_id()} ({confidence:.2f}) [DET]"
                else:
                    label = f"{bb.get_class_id()} [DET]"
                img = add_bb_into_image_with_scale_color(
                    img, bb, thickness=2, label=label, show_scale_in_label=True
                )
                has_bboxes = True
        
        # Add scale color legend to the image if bounding boxes are present
        if has_bboxes:
            img = add_scale_legend_to_image(img, position='top-right', margin=10)
        
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
        Plot the distribution of bounding boxes per object class.
        
        Opens a matplotlib window showing a bar chart of bounding box
        counts for each class in the dataset.
        """
        general_utils.plot_bb_per_classes(self.bb_per_class,
                                          horizontally=False,
                                          rotation=90,
                                          show=True)

    def btn_plot_bb_per_scale_clicked(self):
        """
        Plot the distribution of bounding boxes per COCO scale category.
        
        Opens a matplotlib window showing bar and pie charts of bounding box
        counts for each scale category (small, medium, large) following
        the COCO dataset standard thresholds.
        """
        general_utils.plot_bb_per_scale(self.annot_obj, show=True)

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
