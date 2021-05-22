## Dataset description

The dataset has ~3600 images. These images are of construction workers (mostly from work site). We are focusing on 4 main classes within these images:

hardhat (0)

vest (1)

mask (2)

boots (3)

Note: An image need not have all 4 classes but will have at least one for sure. AN image might have a class more than once.

1. Input images are in **images** folder. These are collected from all the batch students who individually contributed between 50-200 images (at least 50 images of each class). Almost all of them are .jpg/.jpeg files.

2. Bounding Boxes
Bounding boxes are created using annotation tool from here: https://github.com/miki998/YoloV3_Annotation_Tool

Each image contains 1/more annotated regions. An annotated region has 4 parameters: x, y, width, height. (x, y) is the coordinate of the top left corner of the bounding box of width and height. This data is present in a text file (same name as image). The text file has the same number of lines as the number of annotations in the image. First column of each line represents the class it represents. The next 4 are x, y, width and height. Similarly, every image has a corresponding text file with the bounding box information. All these text files are under the **labels** folder.

3. Depth images were created by running the MiDaS repository (https://github.com/intel-isl/MiDaS). These are grayscale images which contains information relating to the distance of the surfaces of scene objects from a viewpoint. These are .png files (with the same name as the .jpg files) available under **depth** folder.

4. Surface plane images were created by running planercnn repo (https://github.com/NVlabs/planercnn). These are plane detection images (again, .png files with the same name as .jpg files).


## Session 14 assignment output

Uploaded **4** output images here (actual output files are very large)

1. https://github.com/paishowstopper/TSAI/tree/main/EVA5/Session14/MiDaS/output_images
2. https://github.com/paishowstopper/TSAI/tree/main/EVA5/Session14/planercnn/output_images


MiDaS full output (screenshot)

![image](https://user-images.githubusercontent.com/26896746/119217284-d13ba400-baf6-11eb-90f4-4af5374a7039.png)

Planercnn full output (screenshot)

![image](https://user-images.githubusercontent.com/26896746/119217314-0647f680-baf7-11eb-8d1e-06c1e8b5ce08.png)
