#!/usr/bin/env python3
"""YOLO object detection"""

import tensorflow.keras as K
import numpy as np
import cv2
import glob
import os


class Yolo:
    """YOLO v3 object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize YOLO"""
        self.model = K.models.load_model(model_path)
        with open(classes_path, "r") as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process YOLO outputs"""
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_h = image_size[0]
        image_w = image_size[1]

        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_h = output.shape[0]
            grid_w = output.shape[1]
            anchor_boxes = output.shape[2]

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))

            c_x = np.arange(grid_w).reshape(1, grid_w, 1)
            c_y = np.arange(grid_h).reshape(grid_h, 1, 1)

            b_x = (1 / (1 + np.exp(-t_x)) + c_x) / grid_w
            b_y = (1 / (1 + np.exp(-t_y)) + c_y) / grid_h

            anchor_w = self.anchors[i, :, 0].reshape(1, 1, anchor_boxes)
            anchor_h = self.anchors[i, :, 1].reshape(1, 1, anchor_boxes)

            b_w = (np.exp(t_w) * anchor_w) / input_w
            b_h = (np.exp(t_h) * anchor_h) / input_h

            x1 = (b_x - b_w / 2) * image_w
            y1 = (b_y - b_h / 2) * image_h
            x2 = (b_x + b_w / 2) * image_w
            y2 = (b_y + b_h / 2) * image_h

            box = np.zeros(output[..., :4].shape)
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2

            boxes.append(box)
            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter YOLO boxes"""
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box, confidence, class_probs in zip(
                boxes, box_confidences, box_class_probs):
            box_score = confidence * class_probs
            box_class = np.argmax(box_score, axis=-1)
            box_class_score = np.max(box_score, axis=-1)
            filtering_mask = box_class_score >= self.class_t

            filtered_boxes.append(box[filtering_mask])
            box_classes.append(box_class[filtering_mask])
            box_scores.append(box_class_score[filtering_mask])

        filtered_boxes = np.concatenate(filtered_boxes)
        box_classes = np.concatenate(box_classes)
        box_scores = np.concatenate(box_scores)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Apply non-max suppression"""
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for cls in np.unique(box_classes):
            idxs = np.where(box_classes == cls)[0]
            cls_boxes = filtered_boxes[idxs]
            cls_scores = box_scores[idxs]
            order = np.argsort(cls_scores)[::-1]

            while order.size > 0:
                best = order[0]

                box_predictions.append(cls_boxes[best])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(cls_scores[best])

                if order.size == 1:
                    break

                current_box = cls_boxes[best]
                other_boxes = cls_boxes[order[1:]]

                x1 = np.maximum(current_box[0], other_boxes[:, 0])
                y1 = np.maximum(current_box[1], other_boxes[:, 1])
                x2 = np.minimum(current_box[2], other_boxes[:, 2])
                y2 = np.minimum(current_box[3], other_boxes[:, 3])

                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)
                intersection = inter_w * inter_h

                current_area = ((current_box[2] - current_box[0]) *
                                (current_box[3] - current_box[1]))
                other_areas = ((other_boxes[:, 2] - other_boxes[:, 0]) *
                               (other_boxes[:, 3] - other_boxes[:, 1]))

                union = current_area + other_areas - intersection
                iou = intersection / union

                keep = np.where(iou <= self.nms_t)[0]
                order = order[keep + 1]

        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """Load images from folder"""
        image_paths = glob.glob(folder_path + "/*")
        images = [cv2.imread(path) for path in image_paths]
        return images, image_paths

    def preprocess_images(self, images):
        """Preprocess images"""
        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]
        image_shapes = []
        pimages = []

        for image in images:
            image_shapes.append(image.shape[:2])
            pimage = cv2.resize(image, (input_w, input_h),
                                interpolation=cv2.INTER_CUBIC)
            pimage = pimage / 255
            pimages.append(pimage)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """Show image boxes"""
        for box, box_class, box_score in zip(boxes, box_classes, box_scores):
            x1, y1, x2, y2 = box.astype(int)
            label = "{} {:.2f}".format(self.class_names[box_class], box_score)

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cv2.putText(image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)

        if key == ord("s"):
            if not os.path.exists("detections"):
                os.makedirs("detections")
            cv2.imwrite(os.path.join("detections", file_name), image)

        cv2.destroyAllWindows()
