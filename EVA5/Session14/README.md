Dataset:
This dataset consists of around 3500 images containing four main classes:

Hard hats
Vests
Masks
Boots
Common settings include construction areas/construction workers, military personnel, traffic policemen etc.
Not all classes are present in an image. Also one image may have many repitions of the same class.
For example, a group of construction workers without helmets, but with vests and boots.

The dataset is available under: https://drive.google.com/drive/u/1/folders/1nD1cdLk5y-rpmtiXH-JeU5vLvLLYVVyp

Explanation:
There are four folders provided:

images
labels
depth
planes
1. Raw images
The raw images are present under the images folder. The images were collected by crowdsourcing and do not follow any particular naming convention.
They are also of varied sizes. There are 3591 images.
These are mostly .jpg files (< 0.5% might be otherwise)

2. Bounding Boxes
A Yolo compatible annotation tool was used to annotate the classes within these images.
These are present under the labels folder as text files. However please note that not all raw images have a corresponding label. There are 3527 labelled text files. A few things to note:

Each image can contain 0 or more annotated regions.
Each annotated region is defined by four main parameters: x, y, width, height
For the rectangular region, (x, y) coordinates refers to top left corner of the bounding box
width and height refer to the width and height of the bounding region. The centroid of the bounding box can be calculated from this if required.
A label file corresponding to an image is a space separated set of numbers. Each line corresponds to one annotated region in the image.
The first column maps to the class of the annotated region (order of the classes is as described above). The other four numbers represent the bounding boxes (ie annotated region), and stand for the x, y, width and height parameters explained earlier. These four numbers should be between 0 and 1.
3. Depth images
Depth images were created using this repo:
https://github.com/intel-isl/MiDaS
There are 3588 depth images, they are present under the 'depth' folder, and are greyscale images
These are .png files, make sure to handle accordingly since the raw images are .jpg
The names are same as that of the corresponding raw images.

4. Planar images
Planes were created using this repo:
https://github.com/NVlabs/planercnn
These are .png files, make sure to handle accordingly since the raw images are .jpg
There are 3545 planar images. The names are same as that of the corresponding raw images.

Note:
This dataset needs to be cleaned up further.

There are a few (<0.5%) png files among the raw images, which need to be removed (These do not have labels ie bounding boxes, nor do they have planar images).
There are a few (<0.5%) label files which are of invalid syntax (the x,y coordinates, or the width/height are > 1). These need to be discarded.
Final cleaned up dataset should only include data where all these three files are present for a raw image: labels text file, depth image and planar image
