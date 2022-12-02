import torch
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import torchvision.ops as ops

import utils

image_size: int = 236
grid_size: int = 3
threshold: float = 0.4

# Every input parameter is expected to be either a list or a tensor with batch_size elements in the first dimension
# This functions plots the images, the ground truth boxes and class ids as well as the predicted boxes and classes
def print_batch_check(image_size, images, target_boxes, target_classes, predicted_boxes, predicted_classes):
    for item_count, (image, item_label_boxes, item_label_classes) in enumerate(
            zip(images, target_boxes, target_classes)):
        # Reverse the normalization and transposition applied when loaded
        img_unnormalized = image / 2 + 0.5
        img_transposed = np.transpose(img_unnormalized, [1, 2, 0])
        fig = plt.figure(figsize=(2, 2))
        ax = fig.add_axes([0, 0, 1, 1])
        plt.imshow(img_transposed)
        for label_count, (boxes, classes) in enumerate(zip(item_label_boxes, item_label_classes)):
            x, y, w, h = boxes * image_size
            box = patches.Rectangle((x - w / 2, y - h / 2), w, h, edgecolor="red", facecolor="none")
            ax.add_patch(box)
            rx, ry = box.get_xy()
            cx = rx + box.get_width() / 2.0
            cy = ry + box.get_height() / 8.0
            class_id = np.argmax(classes).item()
            ax.annotate(
                class_id,
                (cx, cy),
                fontsize=8,
                fontweight="bold",
                color="red",
                ha='center',
                va='center'
            )
        for i, row in enumerate(predicted_boxes[item_count]):
            for j, col in enumerate(row):
                x, y, w, h = (col * image_size)
                box = patches.Rectangle((x - w / 2, y - h / 2), w, h, edgecolor="green", facecolor="none")
                ax.add_patch(box)
                rx, ry = box.get_xy()
                cx = rx + box.get_width() / 2.0
                cy = ry + box.get_height() / 8.0
                class_id = np.argmax(predicted_classes[item_count, i, j]).item()
                ax.annotate(
                    class_id,
                    (cx, cy),
                    fontsize=8,
                    fontweight="bold",
                    color="green",
                    ha='center',
                    va='center'
                )
        plt.axis('off')
        plt.show()


def print_image_check(image_size, image, target_boxes, target_classes, predicted_boxes, predicted_classes):
    # Reverse the normalization and transposition applied when loaded
    img_unnormalized = image / 2 + 0.5
    img_transposed = np.transpose(img_unnormalized, [1, 2, 0])
    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_axes([0, 0, 1, 1])
    plt.imshow(img_transposed)
    for label_count, (boxes, classes) in enumerate(zip(target_boxes, target_classes)):
        x, y, w, h = boxes * image_size
        box = patches.Rectangle((x - w / 2, y - h / 2), w, h, edgecolor="red", facecolor="none")
        ax.add_patch(box)
        rx, ry = box.get_xy()
        cx = rx + box.get_width() / 2.0
        cy = ry + box.get_height() / 8.0
        class_id = np.argmax(classes).item()
        ax.annotate(
            class_id,
            (cx, cy),
            fontsize=8,
            fontweight="bold",
            color="red",
            ha='center',
            va='center'
        )
    for i, row in enumerate(predicted_boxes):
        for j, col in enumerate(row):
            x, y, w, h = (col * image_size)
            box = patches.Rectangle((x - w / 2, y - h / 2), w, h, edgecolor="green", facecolor="none")
            ax.add_patch(box)
            rx, ry = box.get_xy()
            cx = rx + box.get_width() / 2.0
            cy = ry + box.get_height() / 8.0
            class_id = np.argmax(predicted_classes[i, j]).item()
            ax.annotate(
                class_id,
                (cx, cy),
                fontsize=8,
                fontweight="bold",
                color="green",
                ha='center',
                va='center'
            )
    plt.axis('off')
    plt.show()

# Every parameter is of shape (grid_size, grid_size, 4)
# This function interprets the bounding box related outputs of the network as centroid x,y movement and width,height stretch
def bb_activation_to_prediction(boxes_outputs, anchor_boxes, grid_size):
    activated_boxes = torch.tanh(boxes_outputs)
    activated_boxes_centroid = (activated_boxes[:, :, :2] / 2 * grid_size) + anchor_boxes[:, :, :2]
    activated_boxes_width_height = (activated_boxes[:, :, 2:] / 2 + 1) * anchor_boxes[:, :, :2]
    predicted_boxes = torch.cat((activated_boxes_centroid, activated_boxes_width_height), dim=2)
    return predicted_boxes


# The boxes parameter is of shape (grid_size*grid_size, 4)
# This function converts boxes defined as x,y,w,h to x1,y1,x2,y2
def bb_hw_to_corners(boxes):
    x_shift = torch.div(boxes[:, [0, 2]], 2)
    y_shift = torch.div(boxes[:, [1, 3]], 2)
    x_1s = torch.unsqueeze(boxes[:, 0] - x_shift[:, 0], dim=1)
    x_2s = torch.unsqueeze(boxes[:, 2] + x_shift[:, 1], dim=1)
    y_1s = torch.unsqueeze(boxes[:, 1] - y_shift[:, 0], dim=1)
    y_2s = torch.unsqueeze(boxes[:, 3] + y_shift[:, 1], dim=1)
    corner_boxes = torch.cat((x_1s, y_1s, x_2s, y_2s), dim=1)
    return corner_boxes


# The parameter target_boxes is a list of tensors of shape (4)
# This functions converts a list of bounding boxes expressed as x,y,w,h to x1,y1,x2,y2
def boxes_list_to_corners(target_boxes):
    corner_boxes = torch.zeros((len(target_boxes), 4))
    for i, box in enumerate(target_boxes):
        shift = box[2:] / 2
        corner_boxes[i] = torch.cat((box[:2] - shift, box[:2] + shift))
    return corner_boxes


# The predicted_boxes parameter is of shape (grid_size, grid_size, 4) while target_boxes is a list of length
# label_number, each item is then 4 long
# This function calculates the intersection over union coefficient for every predicted_box with respect to every
# target box. The function expects both box groups to be expressed as x,y,w,h
def iou_coefficients(predicted_boxes, target_boxes):
    predicted_boxes = bb_hw_to_corners(predicted_boxes)
    target_boxes = boxes_list_to_corners(target_boxes)
    iou = ops.box_iou(target_boxes, predicted_boxes)
    return iou


# The iou is a (gt_object_number, grid_size*grid_size) tensor containing the iou coefficient of
# every ground truth object with respect to all image's anchor box predictions
def map_to_ground_truth(iou):
    # For every gt object this tells us with which anchor box they overlapped the most
    # This has shape (gt_object_number)
    prior_overlap, prior_index = torch.max(iou, dim=1)
    # For every anchor box this tells us with which gt object they overlapped the most
    # This has shape (grid_size*grid_size)
    gt_overlap, gt_index = torch.max(iou, dim=0)
    # We make sure that every gt object gets assigned at least one anchor box. This value is just high enough to be
    # considered a positive match
    gt_overlap[prior_index] = 1.1
    # We make sure that if an anchor box is the best match for a gt object, the best match for said anchor box becomes
    # with that gt object
    for index, object in enumerate(prior_index):
        gt_index[object] = index
    return gt_overlap, gt_index


def ssd_item_loss(image, pred_classes, pred_boxes, target_classes, target_boxes, anchor_boxes):
    corner_target_boxes = np.divide(target_boxes, image_size)
    activated_boxes = bb_activation_to_prediction(pred_boxes, anchor_boxes, grid_size).flatten(start_dim = 0, end_dim = 1)
    iou_coeff = iou_coefficients(activated_boxes, target_boxes)
    anchor_gt_overlap, anchor_gt_index = map_to_ground_truth(iou_coeff)
    # (gt_object_number, grid_size*grid_size) boolean matrix
    positive_overlap_mask = anchor_gt_overlap > threshold
    positive_index_mask = anchor_gt_index[positive_overlap_mask]
    positive_boxes = activated_boxes[positive_overlap_mask]
    ciao = 2

