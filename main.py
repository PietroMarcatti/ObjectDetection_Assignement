import torch
import torchvision.transforms as transforms
import torch.optim as optim
import utils
from model import Net
from torch.utils.data import DataLoader
from dataset import ObjectDetectionDataSet

# This is the root directory for the model's training files (images and annotations)
train_directory = 'C:/Users/pietr/Downloads/assignment_1/assignment_1/train'
# This is the root directory for the model's testing files (images and annotations)
test_directory = 'C:/Users/pietr/Downloads/assignment_1/assignment_1/test'
# This is the height and width, in pixels, of the images that the model is going to operate on
image_size: int = 236
# This is the number of images fed into the network at each iteration
batch_size: int = 16
# This is the number of classes present in the dataset. It was derived from the annotations files
number_of_classes = 14
# This is the number of equally sized squares that are created for every side of the image to be used as anchor boxes
grid_size: int = 3
# This check will make sure that the device to which tensors are assigned is the best available
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


# We need to define a new collate function that allows us to have different length tensor for each sample in the batch.
# This is needed because samples have different amount of boxes/classes to detect
def variable_size_collate_fn(batch):
    batch_images = torch.empty(batch_size, 3, image_size, image_size)
    batch_boxes_label = []
    batch_classes_label = []
    for item_counter, (image, label_boxes, label_classes) in enumerate(batch):
        batch_images[item_counter] = image
        batch_boxes_label.append(label_boxes)
        batch_classes_label.append(label_classes)
    return batch_images, batch_boxes_label, batch_classes_label


if __name__ == '__main__':
    imageTransformation = transforms.Compose([transforms.ToTensor(),
                                              transforms.Resize(size=(image_size, image_size)),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                              ])
    train_dataset = ObjectDetectionDataSet(root=train_directory,
                                           transform=imageTransformation,
                                           imageSize=image_size,
                                           numberOfClasses=number_of_classes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4,
                              collate_fn=variable_size_collate_fn)
    test_dataset = ObjectDetectionDataSet(root=test_directory,
                                          transform=imageTransformation,
                                          imageSize=image_size,
                                          numberOfClasses=number_of_classes)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=4,
                                             collate_fn=variable_size_collate_fn)

    # Anchor boxes are the base guesses of our model, they are going to a grid_size x grid_size of equal sized squares
    # Each anchor box is defined by its centroid (x, y), width and height, hence the tensor (grid_size, grid_size, 4)


    net = Net(number_of_classes).to(device)

    data_iterator = iter(train_loader)
    images_batch, boxes_label_batch, classes_label_batch = next(data_iterator)
    classes_placeholder = torch.zeros((batch_size, grid_size, grid_size, 1))

    # utils.print_batch_check(image_size, images_batch, boxes_label_batch, classes_label_batch,
    #                         expanded_anchor_boxes, classes_placeholder)
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.01)
    criterion = utils.ssd_loss
    loss_history = []

    for epoch in range(35):
        running_loss = 0.0
        for batch_number, batch in enumerate(train_loader):
            batch_images, batch_boxes, batch_classes = batch
            batch_images = batch_images.to(device)

            optimizer.zero_grad()
            output = net(batch_images)
            output = output.permute(0, 2, 3, 1)
            box_prediction = output[:, :, :, :4]
            class_prediction = output[:, :, :, 4:]
            loss = criterion(class_prediction, box_prediction, batch_classes, batch_boxes)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            running_loss += loss.item()
            if batch_number % 250 == 249:  # Printing the runnin loss every 500 mini-batches
                print(f"[epoch: {epoch + 1}, mini-batch: {batch_number + 1}] loss: {(running_loss / 250):.3f}")
                running_loss = 0.0
    print('Finished Training!')
