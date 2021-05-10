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

def find_misclassified(model, testloader, numSamples=25):
    incorrect_indexes = {}  # {23: {'actual': 1, 'predicted': 4}}
    model.eval()
    count = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            for sampleno in range(data.shape[0]):
                if (target[sampleno] != pred[sampleno]):
                    count += 1
                    # print("Index=", sampleno, ", Actual=", target[sampleno].cpu().numpy(), ", Predicted: ", pred[sampleno].cpu().numpy()[0])
                    incorrect_indexes[sampleno] = {'actual': target[sampleno].cpu().numpy(),
                                                   'predicted': pred[sampleno].cpu().numpy()[0],
                                                   'data': data[sampleno].cpu().numpy()}

            if count == numSamples:
                break
    return incorrect_indexes

# test and training curves