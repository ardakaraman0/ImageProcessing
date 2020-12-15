#Arda Karaman 2237568


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import sys

filename = sys.argv[1]

source = mpimg.imread(filename)

sobelV = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], np.float32)

sobelH = np.array([[1, 2, 1],
                  [0, 0, 0],
                  [-1,-2,-1]], np.float32)


def grayscale(source):
    """
    takes the source image and returns the grayscale version of it.
    """
    return np.dot(source[...,:3], [0.2989, 0.5870, 0.1140])
    
def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def convolution(source, filt):
    """
    convolution function
    """
    imgHeight, imgWidth = source.shape[0], source.shape[1]
    kerHeight, kerWidth = (filt.shape[0] - 1) // 2, (filt.shape[1] - 1) // 2

    filteredImage = np.zeros_like(source, dtype=float)

    for x in np.arange(kerHeight, imgHeight-kerHeight):
        for y in np.arange(kerWidth, imgWidth-kerWidth):
            finalVal = 0
            
            for i in np.arange(-kerHeight, kerHeight+1):
                for j in np.arange(-kerWidth, kerWidth+1):

                    img = source[x+i, y+j]
                    ker = filt[kerHeight+i, kerWidth+j]
                    finalVal += img * ker

            filteredImage[x, y] = finalVal  

    return filteredImage

def sobel(source, vert, hori):
    """
    sobel filtering function (gradient intensity)
    """
    sobV = convolution(source, vert)
    sobH = convolution(source, hori)

    output = np.hypot(sobV, sobH)
    output = output / output.max() * 255
    theta = np.arctan2(sobH, sobV)

    return (output, theta)

def nonmaxima(image, theta):
    height, width = image.shape[0], image.shape[1]
    output = np.zeros((height,width), dtype=np.int32)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1,height-1):
        for j in range(1,width-1):
            try:
                q = 255
                r = 255
                
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = image[i, j+1]
                    r = image[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = image[i+1, j-1]
                    r = image[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = image[i+1, j]
                    r = image[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = image[i-1, j-1]
                    r = image[i+1, j+1]

                if (image[i,j] >= q) and (image[i,j] >= r):
                    output[i,j] = image[i,j]
                else:
                    output[i,j] = 0

            except IndexError as e:
                pass
    
    return output

def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(75)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)

def hysteresis(img, weak=25, strong=255):
    M, N = img.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


# Changing the kernel size and sigma
kernel = gaussian_kernel(7, 5)

# You can comment out some of the lines to see the images before its finalized
source = grayscale(source)
blurred = convolution(source, kernel)
sobelled, theta = sobel(blurred, sobelV, sobelH)
maxima = nonmaxima(sobelled, theta)
thresholded, weak, strong = threshold(maxima)
finalimg = hysteresis(thresholded, weak, strong)


# For seeing the whole image
# fig, axs = plt.subplots(3,2)
# plt.subplots_adjust(wspace=0.5, hspace=0.5)
# fig.suptitle('Part 2')
# fig.set_size_inches(18.5, 10.5)

# axs[0][0].imshow(source, cmap=plt.get_cmap('gray'))
# axs[0][1].imshow(blurred, cmap=plt.get_cmap('gray'))
# axs[1][0].imshow(sobelled, cmap=plt.get_cmap('gray'))
# axs[1][1].imshow(maxima, cmap=plt.get_cmap('gray'))
# axs[2][0].imshow(thresholded, cmap=plt.get_cmap('gray'))
# axs[2][1].imshow(finalimg, cmap=plt.get_cmap('gray'))

# axs[0,0].set_title("grayscale")
# axs[0,1].set_title("gaussian blur")
# axs[1,0].set_title("sobel filter")
# axs[1,1].set_title("non maxima suppression")
# axs[2,0].set_title("double threshold")
# axs[2,1].set_title("hysteresis")


plt.figure(figsize=(8,10))
plt.imshow(finalimg, cmap=plt.get_cmap('gray'))

plt.savefig("canny-edge-detected" + filename)
plt.show()