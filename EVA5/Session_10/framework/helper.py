import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
from torchsummary import summary
from torchvision.utils import make_grid
from gradcam import GradCAM, visualize_cam

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def DisplayImages(dataloader, classes, count=4):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(count)))

def DisplayModelSummary(model, input_size):
    print(summary(model, input_size=input_size))

def DisplayClassAccuracy(model, dataloader, classes, device):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data, label in dataloader:
            images, labels = data.to(device), label.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

def DisplayGradCamImages(model, model_type, layer, dataloader, classes, device, count=10):
    
    gradcam = GradCAM.from_config(**dict(model_type=model_type, arch=model, layer_name=layer))
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    outputs = model(images.to(device))
    _, predicted = torch.max(outputs.data, 1)

    for i in range(count):
        imagestodisplay = []
        mask, _ = gradcam(images[i][np.newaxis, :].to(device))
        heatmap, result = visualize_cam(mask, images[i][np.newaxis, :])
        imagestodisplay.extend([images[i].cpu(), heatmap, result])
        grid_image = make_grid(imagestodisplay, nrow=3)
        plt.figure(figsize=(20, 20))
        plt.imshow(np.transpose(grid_image, (1, 2, 0)))
        plt.show()
        print(f"Prediction : {classes[predicted[i]]}, Actual : {classes[labels[i]]}")


def GetMisclassifiedImageIndexes(model, dataloader, device, count=25):
    
    misclassified_indexes = {}
    model.eval()
    index = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            predicted = output.argmax(dim=1, keepdim=True)
            for i in range(data.shape[0]):
                if (target[i] != predicted[i]):
                    index += 1
                    misclassified_indexes[i] = {'actual': target[i].cpu().numpy(),
                                                'predicted': predicted[i].cpu().numpy()[0],
                                                'data': data[i].cpu().numpy()}
            if index == count:
                break
    return misclassified_indexes

def DisplayMisclassifiedGradCamImages(model, model_type, layer, misclassified_indexes, device, classes):
    
    gradcam = GradCAM.from_config(**dict(model_type=model_type, arch=model, layer_name=layer))

    x, y = 0, 0
    fig, axs = plt.subplots(5, 5, figsize=(20, 20))
    plt.setp(axs, xticks=[], yticks=[])
    fig.subplots_adjust(wspace=0.7)
    images = list(misclassified_indexes.items())[:25]
    for index, results in images:
        img = results['data']
        img = torch.from_numpy(img)

        actual_class = classes[results['actual']]
        predicted_class = classes[results['predicted']]

        mask, _ = gradcam(img[np.newaxis, :].to(device))
        heatmap, result = visualize_cam(mask, img[np.newaxis, :])
        result = np.transpose(result.cpu().numpy(), (1, 2, 0))

        axs[x, y].imshow(result)
        axs[x, y].set_title('Actual Class:' + str(actual_class) + "\nPredicted class: " + str(predicted_class))

        if y == 4:
            x += 1
            y = 0
        else:
            y += 1

def PlotCurves(train_accuracies, test_accuracies):
    fig, ax = plt.subplots()
    ax.plot(range(0, len(train_accuracies)), train_accuracies, label='Training Accuracy', color='blue')
    ax.plot(range(0, len(test_accuracies)), test_accuracies, label='Test Accuracy', color = 'red')
    legend = ax.legend(loc='center right', fontsize='x-large')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Training and Test Accuracy')
    plt.show()