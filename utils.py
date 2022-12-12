import torch
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import torchvision.ops as ops


image_size: int = 236
grid_size: int = 3
threshold: float = 0.4
number_of_classes = 14
class_criterion = torch.nn.BCEWithLogitsLoss()
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
cell_size = 1 / grid_size

# Anchor boxes are the base guesses of our model, they are going to a grid_size x grid_size of equal sized squares
# Each anchor box is defined by its centroid (x, y), width and height, hence the tensor (grid_size, grid_size, 4)
anchor_boxes = torch.tensor(
            [[[cell_size * (j + 0.5), cell_size * (i + 0.5), cell_size, cell_size] for j in range(grid_size)] for i in range(grid_size)],
            dtype=torch.float32)
anchor_boxes = anchor_boxes.to(device)

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


# Every input parameter is expected to be either a list or a tensor with batch_size elements in the first dimension
# This function is the single image version of the above printing function
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
    activated_boxes_width_height = (activated_boxes[:, :, 2:] / 2 + 1) * anchor_boxes[:, :, 2:]
    predicted_boxes = torch.cat((activated_boxes_centroid, activated_boxes_width_height), dim=2)
    return predicted_boxes


# The boxes parameter is of shape (grid_size*grid_size, 4)
# This function converts a list of boxes defined as x,y,w,h to x1,y1,x2,y2, this is used for the target boxes
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
    target_boxes = boxes_list_to_corners(target_boxes).to(device)
    iou = ops.box_iou(target_boxes, predicted_boxes)
    return iou, predicted_boxes


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
    # considered the best positive match
    gt_overlap[prior_index] = 1.1
    # We make sure that if an anchor box is the best match for a gt object, the best match for said anchor box becomes
    # with that gt object
    for index, object in enumerate(prior_index):
        gt_index[object] = index
    return gt_overlap, gt_index


# This function expect a single image's predicted classes, boxes and target image and boxes
# This function just interprets the output of our network in a way that it can be compared with the ground truths
def ssd_item_evaluation(pred_classes, pred_boxes, target_classes, target_boxes):
    corner_target_boxes = target_boxes.cpu()
    activated_boxes = bb_activation_to_prediction(pred_boxes, anchor_boxes, grid_size).flatten(start_dim=0, end_dim=1)
    iou_coeff, activated_boxes = iou_coefficients(activated_boxes, target_boxes)
    anchor_gt_overlap, anchor_gt_index = map_to_ground_truth(iou_coeff)
    # (gt_object_number, grid_size*grid_size) boolean matrix
    positive_overlap_mask = anchor_gt_overlap > threshold
    positive_index_mask = torch.nonzero(positive_overlap_mask)[:, 0]
    predicted_boxes = activated_boxes[positive_index_mask]
    gt_boxes = corner_target_boxes[anchor_gt_index.cpu()]
    targ_boxes = gt_boxes[positive_index_mask.cpu()].to(device)

    target_class_ids = torch.argmax(target_classes, dim=1)
    gt_class = target_class_ids[anchor_gt_index.cpu()]
    gt_class[~positive_overlap_mask.cpu()] = number_of_classes
    gt_class = torch.eye(number_of_classes + 1)[gt_class.cpu()]
    targ_classes = gt_class.to(device)
    predicted_classes = pred_classes.flatten(start_dim=0, end_dim=1)

    return predicted_boxes, targ_boxes, predicted_classes ,targ_classes

# This function expects a single image's predicted classes, boxes and target image and boxes
# This function interprets the output of the network so that we can calculate the loss on this image
def ssd_item_loss(pred_classes, pred_boxes, target_classes, target_boxes):
    # The ground truth boxes are expressed as x,y,w,h
    corner_target_boxes = target_boxes.cpu()
    # We modify the anchor boxes with the x,y shift and w,h stretch/shrink obtained from the network
    activated_boxes = bb_activation_to_prediction(pred_boxes, anchor_boxes, grid_size).flatten(start_dim=0, end_dim=1)
    # We calculate the iou coefficients between the activated anchor boxes and the target boxes
    iou_coeff, activated_boxes = iou_coefficients(activated_boxes, target_boxes)
    # We map each predicted box to a target box finding the best match
    anchor_gt_overlap, anchor_gt_index = map_to_ground_truth(iou_coeff)
    # We consider a mapping between a predicted box and target box to be positive only if a certain iou is reached
    positive_overlap_mask = anchor_gt_overlap > threshold
    # We create an index mask for the positive matches
    positive_index_mask = torch.nonzero(positive_overlap_mask)[:, 0]
    # We identify the boxes that have a positive match
    positive_boxes = activated_boxes[positive_index_mask]
    # We create a tensor with the target box that best matches each anchor box
    gt_boxes = corner_target_boxes[anchor_gt_index.cpu()]
    # We calculate the L1 loss for the boxes between the positive matches and the corresponding target boxes
    box_loss = (positive_boxes - gt_boxes[positive_index_mask.cpu()].to(device)).abs().mean()

    # We get the ids (0-13) of the target classes
    target_class_ids = torch.argmax(target_classes, dim=1)
    # We create a list of the associated target class for each box (according to the class of the box with which they most overlap)
    gt_class = target_class_ids[anchor_gt_index.cpu()]
    # For those anchor boxes that do not have a positive match we associate a special class "background" id 14
    gt_class[~positive_overlap_mask.cpu()] = number_of_classes
    gt_class = torch.eye(number_of_classes+1)[gt_class.cpu()]
    # We remove the last element from the expected classes so that the loss function will have to try to minimize all the values
    gt_class = gt_class[:, :-1].to(device)
    pred_classes = pred_classes.flatten(start_dim=0, end_dim=1)[:, :-1]
    # We calculate the loss for the classification task with the binary cross entropy with logits
    class_loss = class_criterion(pred_classes, gt_class)
    return box_loss, 3*class_loss

# This function will call the single image loss function calculation and sum it for the entire batch, allowing for "seameless"
# batch operation
def ssd_loss(batch_pred_classes, batch_pred_boxes, batch_target_classes, batch_target_boxes):
    localization_loss = 0.
    classification_loss = 0.
    for pred_classes, pred_boxes, target_classes, target_boxes in zip(batch_pred_classes, batch_pred_boxes, batch_target_classes, batch_target_boxes):
        target_boxes = target_boxes.to(device)
        target_classes = target_classes.to(device)
        loc_loss, class_loss = ssd_item_loss(pred_classes, pred_boxes, target_classes, target_boxes)
        localization_loss += loc_loss
        classification_loss += class_loss
    return localization_loss + classification_loss
