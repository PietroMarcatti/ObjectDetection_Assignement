import torch
import os
import json
import numpy as np
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset


class ObjectDetectionDataSet(Dataset):
    def __init__(self, root, transform, imageSize, numberOfClasses):
        self.main_dir = root
        self.img_dir = os.path.join(self.main_dir, 'images')
        self.json_dir = os.path.join(self.main_dir, 'annotations')
        self.transform = transform
        self.imageSize = imageSize
        self.numberOfClasses = numberOfClasses
        all_imgs = os.listdir(self.img_dir)
        self.total_imgs = natsorted(all_imgs)
        all_json = os.listdir(self.json_dir)
        self.total_json = natsorted(all_json)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        # Image path
        img_loc = os.path.join(self.img_dir, self.total_imgs[idx])
        # Load Image
        image = Image.open(img_loc).convert("RGB")
        # Get size for resizing boxes keeping the ratio
        imageWidth, imageHeigth = image.size
        # Apply transformation (Tensorization, Resizing, Normalization)
        image = self.transform(image)
        # Calculating aspect ratio of image
        ratio = torch.tensor(
            [self.imageSize / imageWidth, self.imageSize / imageHeigth]).repeat(2)

        # JSON Annotation file path
        json_loc = os.path.join(self.json_dir, self.total_json[idx])
        # Open and load JSON file
        f = open(json_loc)
        jsonf = json.load(f)
        # Each file has (number of item + 2) keys, then we deduce the number of labels for the sample
        label_number = len(jsonf.keys()) - 2
        # The label tensor for the image is organized as follows
        # 0-3 for the bounding box coordinates (x,y,w,h)
        # 4-numberOfClasses+1 for the one-hot vector used in the classification task
        label_boxes = torch.zeros((label_number, 4))
        label_classes = torch.zeros((label_number, self.numberOfClasses))
        # Iterate through the items to populate the label tensor
        for i in range(1, label_number + 1):
            # Get the bounding boxes coordinates from the annotation file
            x_min, y_min, x_max, y_max = np.multiply(jsonf['item' + str(i)]['bounding_box'], ratio)
            # Calculate the centroid position
            centroid_x, centroid_y = ((x_min + x_max) / 2, (y_min + y_max) / 2)
            # Calculate the height and width of the bounding box
            bbox_w, bbox_h = ((abs(x_max - x_min)), (abs(y_max - y_min)))
            # Create a one-hot vector which is active for the class of the target
            label_classes[i - 1] = torch.nn.functional.one_hot(torch.tensor([jsonf['item' + str(i)]['category_id']]), self.numberOfClasses)[0]
            # Put together the label vector
            label_boxes[i - 1] = torch.div(torch.tensor([centroid_x, centroid_y, bbox_w, bbox_h]), self.imageSize)

        return image, label_boxes, label_classes
