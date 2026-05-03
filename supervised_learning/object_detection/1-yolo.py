#!/usr/bin/env python3
"""YOLO class"""

import tensorflow.keras as K
import numpy as np


class Yolo:
    """YOLO v3 model"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize YOLO"""
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process outputs"""
        boxes = []
        box_confidences = []
        box_class_probs = []

        input_h = int(self.model.input.shape[1])
        input_w = int(self.model.input.shape[2])
        image_h, image_w = image_size

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            c_x = np.arange(grid_w)
            c_y = np.arange(grid_h)
            c_x, c_y = np.meshgrid(c_x, c_y)

            c_x = np.expand_dims(c_x, axis=-1)
            c_y = np.expand_dims(c_y, axis=-1)

            b_x = (1 / (1 + np.exp(-t_x)) + c_x) / grid_w
            b_y = (1 / (1 + np.exp(-t_y)) + c_y) / grid_h

            anchors = self.anchors[i]
            p_w = anchors[:, 0].reshape(1, 1, anchor_boxes)
            p_h = anchors[:, 1].reshape(1, 1, anchor_boxes)

            b_w = (np.exp(t_w) * p_w) / input_w
            b_h = (np.exp(t_h) * p_h) / input_h

            box = np.zeros(output[..., :4].shape)

            box[..., 0] = (b_x - b_w / 2) * image_w
            box[..., 1] = (b_y - b_h / 2) * image_h
            box[..., 2] = (b_x + b_w / 2) * image_w
            box[..., 3] = (b_y + b_h / 2) * image_h

            boxes.append(box)
            box_confidences.append(1 / (1 + np.exp(-output[..., 4:5])))
            box_class_probs.append(1 / (1 + np.exp(-output[..., 5:])))

        return boxes, box_confidences, box_class_probs
