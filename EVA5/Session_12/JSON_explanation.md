The generated json file has 5 main sections:

{
  "_via_settings": {settings_for_ui},
  "_via_img_metadata": {annotations_per_image},
  "_via_attributes": {annotations_metadata},
  "_via_image_id_list": [image_list],
  "_via_data_format_version": version
}  

1. _via_img_metadata
Details of all the annotations for each image file.

Sample:

"600-02217157en_Masterfile.jpg26475":
{"filename":"600-02217157en_Masterfile.jpg",
"size":26475,
"regions":[{"shape_attributes":{"name":"rect","x":27,"y":360,"width":239,"height":77},
            "region_attributes":{"name":"boots","type":"unknown","image_quality":{"good":true,"frontal":true,"good_illumination":true}}},
            {"shape_attributes":{"name":"rect","x":87,"y":105,"width":131,"height":122},
            "region_attributes":{"name":"vest","type":"unknown","image_quality":{"good":true,"frontal":true,"good_illumination":true}}}],
"file_attributes":{"caption":"","public_domain":"no","image_url":""}}

filename: Name of the file
size: image_height x image_width
file_attributes: File metadata - any captions, public domain or not and image url.
**regions: shape_attributes, region_attributes**
Describes 1/more annotated regions of the image. 
shape_attributes - Describes the shape of the annotation (For example, Rectangle) along with the properties of the shape (height, width, (x,y) coordinates - Top-left corner)
region_attributes - Describes the details of the annotation - Name (e.g.- Boots, vest, etc.), type (human, bird, cup or unknown), whether frontal and/or blurred as we entered on the tool

2. _via_settings
List of UI settings - Height/Width/Font features of the editor, Height/width of the image grid, color/thickness/shape/etc. of the annotations made on the image, font/color/name/etc. of the annotation details entered for each image, project name (this will be the default name of the json file created from the tool)

3. _via_attributes
List of attributes we can assign for each of the image. For example, while annotating an image using the tool, we have options to specify the type of the object (human, bird, cup or unknown), whether the image is blurred, whether it's a frontal view, etc.

4. _via_image_id_list
List of images uploaded for annotation

5. _via_data_format_version
Data format version (2.0.10)
