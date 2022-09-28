# Arda Karaman 2237568

import numpy as np
import cv2
from skimage import morphology
from matplotlib import pyplot as plt
import argparse

# Parsing the input fileName
parser = argparse.ArgumentParser(description='The name of the picture.')
parser.add_argument('input', type=str, help='The name of the image')
args = parser.parse_args()
imageName = args.input

# Specifying the parameters with respect to the image
if imageName[imageName.find('/')+1:] == 'A1.png':
    img = cv2.imread('THE3-Images/A1.png')

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5,5),np.uint8)

    final = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    ret, _ = cv2.connectedComponents(final)

    print("The number of flying jets in image A1 is " + str(ret-1))
    plt.imshow(final, cmap="gray")
    plt.imsave("part1_A1.png", final)
    plt.show()

elif imageName[imageName.find('/')+1:] == 'A2.png':
    img = cv2.imread('THE3-Images/A2.png')

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5,5),np.uint8)

    final = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    ret, _ = cv2.connectedComponents(final)

    print("The number of flying jets in image A2 is " + str(ret-1))

    plt.imshow(final, cmap="gray")
    plt.imsave("part1_A2.png", final)
    plt.show()


elif imageName[imageName.find('/')+1:] == 'A3.png':
    img = cv2.imread('THE3-Images/A3.png')

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 69, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5,5),np.uint8)

    final = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    final = cv2.erode(final, kernel, iterations=2)

    ret, _ = cv2.connectedComponents(final)

    print("The number of flying jets in image A3 is " + str(ret-1))
    plt.imshow(final, cmap="gray")
    plt.imsave("part1_A3.png", final)
    plt.show()


elif imageName[imageName.find('/')+1:] == 'A4.png':
    img = cv2.imread('THE3-Images/A4.png')

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3,3),np.uint8)
    er = cv2.erode(thresh, kernel, iterations=1)

    kernel = np.ones((5,5),np.uint8)
    final = cv2.morphologyEx(er, cv2.MORPH_CLOSE, kernel)

    arr = np.array(final)
    arr = morphology.remove_small_holes(arr, 10)

    ret, _ = cv2.connectedComponents(final)

    print("The number of flying jets in image A4 is " + str(ret-1))

    plt.imshow(arr, cmap="gray")
    plt.imsave("part1_A4.png", final)
    plt.show()

elif imageName[imageName.find('/')+1:] == 'A5.png':
    img = cv2.imread('THE3-Images/A5.png')

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)


    kernel = np.ones((2,2),np.uint8)
    er = cv2.erode(thresh, kernel, iterations=20)

    kernel = np.ones((5,5),np.uint8)
    final = cv2.dilate(er, kernel, iterations=12)

    ret, _ = cv2.connectedComponents(final)

    print("The number of flying jets in image A5 is " + str(ret-1))

    plt.imshow(final, cmap="gray")
    plt.imsave("part1_A5.png", final)
    plt.show()

elif imageName[imageName.find('/')+1:] == 'A6.png':
    img = cv2.imread('THE3-Images/A6.png')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((4,4),np.uint8)
    er = cv2.erode(thresh, kernel, iterations=2)

    kernel = np.ones((5,5),np.uint8)
    final = cv2.morphologyEx(er, cv2.MORPH_CLOSE, kernel)

    final = cv2.dilate(er, kernel, iterations=10)

    ret, _ = cv2.connectedComponents(final)

    print("The number of flying jets in image A6 is " + str(ret-3))
    # :( 

    plt.imshow(final, cmap="gray")
    plt.imsave("part1_A6.png", final)
    plt.show()
else:
    print("Erroneous input file name.")



