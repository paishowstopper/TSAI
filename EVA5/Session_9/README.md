**Albumentation File Code**

Albumentations Used:
    **Standard**:
        _HorizontalFlip
        Normalize
        ToTensor_
    **Additional**:
        _Rotate
        RandomCrop
        VerticalFlip_
    **Additional (Commented out for the final run)**:
        _Blur, ChannelShuffle, InvertImg, Cutout_

References:
https://albumentations.ai/docs/api_reference/augmentations/transforms/
https://albumentations.ai/docs/api_reference/pytorch/transforms/

    def AlbumentationTrainTransform(self):
        tf = tc.Compose([ta.HorizontalFlip(p=0.5),
                            ta.Rotate(limit=(-20, 20)),
                            # ta.VerticalFlip(p=0.5),
                            # ta.Cutout(num_holes=3, max_h_size=8, max_w_size=8, p=0.5),
                            # ta.Blur(),
                            # ta.ChannelShuffle(),
                            # ta.InvertImg(),
                            ta.RandomCrop(height=30, width=30, p=5.0),
                            ta.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                            tp.ToTensor()
                            ])
        return lambda img: tf(image = np.array(img))["image"]

    def AlbumentationTestTransform(self):
        tf = tc.Compose([ta.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                        tp.ToTensor()
                        # tp.ToTensor(dict(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.2435, 0.2616)))
                        ])
        return lambda img: tf(image = np.array(img))["image"]"]

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

**Final test accuracy:** 88%

