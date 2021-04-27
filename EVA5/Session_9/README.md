**Albumentation File Code**

#https://albumentations.ai/docs/api_reference/augmentations/transforms/
#https://albumentations.ai/docs/api_reference/pytorch/transforms/

#Normalize is set as ToTensor parameter - dict(mean, std)
def AlbumentationTrainTransform(self):
    atf = tc.Compose([ta.HorizontalFlip(),
                        # ta.Blur(),
                        # ta.ChannelShuffle(),
                        # ta.InvertImg(),
                        # ta.Rotate(),
                        ta.RandomCrop(height=30, width=30, p=5.0),
                        # ta.Cutout(1, 8, 8, [0.4914, 0.4822, 0.4465]),
                        tp.ToTensor(dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
                        ])
    return lambda img: atf(image = np.array(img))["image"]

def AlbumentationTestTransform(self):
    atf = tc.Compose([tp.ToTensor(dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))])
    return lambda img: atf(image = np.array(img))["image"]

**GradCam Module's code**

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

**Notebook link:** https://github.com/paishowstopper/TSAI/blob/main/EVA5/Session_9/Session_9.ipynb

**Framework Link:** https://github.com/paishowstopper/TSAI/new/main/EVA5/Session_9/framework

**Final test accuracy:** 84% (Tested with multiple albumentations and different batch sizes but could not achieve the target accuracy (Spent over a day just training this code without success). Achieved 86% once but could not replicate it.)

