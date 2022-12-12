import torch
import torchvision.transforms as transforms
import torch.optim as optim
import utils
import matplotlib.pyplot as plt
from model import Net
from torch.utils.data import DataLoader
from dataset import ObjectDetectionDataSet
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


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

    net = Net(number_of_classes).to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.01)
    criterion = utils.ssd_loss

    for epoch in range(1):
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
            running_loss += loss.item()
            if batch_number % 250 == 249:  # Printing the runnin loss every 500 mini-batches
                print(f"[epoch: {epoch + 1}, mini-batch: {batch_number + 1}] loss: {(running_loss / 250):.3f}")
                running_loss = 0.0
    print('Finished Training!')

    # Model "evaluation", doesn't work because the model didn't really learn, lol
    correct = 0
    total = 0

    gold_test = []
    pred_test = []

    with torch.no_grad():
        for batch in testloader:
            images, item_boxes, item_labels = batch
            images = images.to(device)
            outputs = net(images)
            outputs = outputs.permute(0, 2, 3, 1)
            for output, boxes, labels in zip(outputs, item_boxes, item_labels):
                p_box, t_box, p_clas, t_clas = utils.ssd_item_evaluation(output[:, :, 4:],output[:, :, :4], labels, boxes)
                p_clas = p_clas.detach().cpu()
                p_clas = torch.argmax(p_clas, 1)
                t_clas = torch.argmax(t_clas.detach().cpu(),1)
                t_mask = t_clas != 14
                gold_test += t_clas[t_mask].tolist()
                pred_test += p_clas[t_mask].tolist()
                total += p_clas.size(0)
                correct += (p_clas == t_clas).sum().item()
    classes = [f'{i}' for i in range(14)]
    print(classification_report(gold_test, pred_test, target_names=classes))

    confmat = confusion_matrix(gold_test, pred_test)

    cm = plt.imshow(confmat, cmap="Greens")
    cbar = plt.colorbar(cm)

    plt.xticks(range(len(classes)), classes, rotation=90)
    plt.yticks(range(len(classes)), classes, rotation=0)

    plt.show()

