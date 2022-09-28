# ImageProcessing
This repo contains my own image processing/computer vision projects and code examples from my courses.

## Histogram Matching
Histogram matching or histogram specification is the transformation of an image so that its histogram matches another specified image's histogram. Implemented
without using OpenCV or sci-kit image frameworks. 

Can be used as:
```bash
python <source_image_file> <reference_image_file>
```

## Canny Edge Detector
The Canny edge detector is an edge detection operator that uses a multi-stage algorithm to detect a wide range of edges in images. 
The algorithm is implemented without using OpenCV or sci-kit image frameworks. 

<<<<<<< .merge_file_fdknPJ
# Object Counting using Morphological operations
A collection of non-linear operations related to the shape or morphology of features in an image is known as Morphological Operation in Image Processing. 
A threshold of the picture was taken. 
The threshold parameters varied according to the pictures.
 
The threshold function goes like this; 
- For every pixel, the same threshold value is applied. 
- If the pixel value is smaller than the threshold, it is set to 0, otherwise it is set to the maximum value determined by the program.

After the threshold function is applied, a set of morphological operations were applied. 
The morphological operations were needed to count the images in a more convenient way. 
The last part of the algorithm which was to count the objects, was done by the ultimate erosion function. 
Lastly, the white pixels left in the image are counted to print the objects contained in the image.

# K-Means & Watershed Algorithms
Segmentation task done by using both algorithms.
=======
Can be used as:
```bash
python <image_file> 
```

### Requirements
- [Pillow](https://pillow.readthedocs.io/en/stable/)
- [Matplotlib](https://matplotlib.org/)
- [NumPy](https://numpy.org/)
>>>>>>> .merge_file_JG3hpg
