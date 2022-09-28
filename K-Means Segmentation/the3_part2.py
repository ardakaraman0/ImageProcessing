# Arda Karaman 2237568

import numpy as np
import cv2
from skimage.color import label2rgb
import imutils
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import argparse

def segmentation_function(inp, iname ,segType='kmeans'):

    if segType=='watershed':
        shifted = cv2.pyrMeanShiftFiltering(inp, 21, 51)

        gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        D = ndimage.distance_transform_edt(thresh)
        localMax = peak_local_max(D, indices=False, min_distance=20,
            labels=thresh)

        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=thresh)

        for label in np.unique(labels):
            if label == 0:
                continue

            mask = np.zeros(gray.shape, dtype="uint8")
            mask[labels == label] = 255
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)
            ((x, y), r) = cv2.minEnclosingCircle(c)
            cv2.circle(inp, (int(x), int(y)), int(r), (0, 255, 0), 2)
            cv2.putText(inp, "#{}".format(label), (int(x) - 10, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imwrite("the3_"+iname+"_output.jpg", inp)

    elif (segType=='kmeans'):
        Z = inp.reshape((-1,3))
        Z = np.float32(Z)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        # Change the number of clusters by changing K
        K = 3
        
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((inp.shape))

        res2 = cv2.GaussianBlur(res2, (5,5), 0)
        cannyImage = cv2.Canny(res2,100,200)

        _, contours, hierarchy = cv2.findContours(cannyImage,  
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))

        cv2.drawContours(res2, contours, -1, (0, 0, 255), 3)

        cv2.imwrite("the3_"+iname+"_output.jpg", res2)
    else:
        return



# Parsing the input fileName
parser = argparse.ArgumentParser(description='The name of the picture.')
parser.add_argument('input', type=str, help='The name of the image')
args = parser.parse_args()
imageName = args.input


# Specifying the parameters with respect to the image
if imageName[imageName.find('/')+1:] == 'B1.jpg':
    img = cv2.imread(imageName)
    segmentation_function(img, "B1")
elif imageName[imageName.find('/')+1:] == 'B2.jpg':
    img = cv2.imread(imageName)
    segmentation_function(img, "B2")
elif imageName[imageName.find('/')+1:] == 'B3.jpg':
    img = cv2.imread(imageName)
    segmentation_function(img, "B3")
elif imageName[imageName.find('/')+1:] == 'B4.jpg':
    img = cv2.imread(imageName)
    segmentation_function(img, "B4")
elif imageName[imageName.find('/')+1:] == 'B5.jpg':
    img = cv2.imread(imageName)
    segmentation_function(img, "B5")
else:
    print("Erroneous input file name.")



