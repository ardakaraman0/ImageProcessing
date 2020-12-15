#Arda Karaman 2237568

# Histogram matching 

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import os
import sys


def histogram_matching(source,reference):
    """
    Adjusts the pixel values of r,g,b image that the source image's histogram
    matches that of the reference image.
    """
    shape = source.shape
    source_raveled = source.ravel()
    reference_raveled = reference.ravel()

    source_pixel_val_arr, reverse, source_pixel_counts = np.unique(source_raveled, 
                                                                    return_inverse=True, return_counts=True)
    reference_pixel_val_arr, reference_pixel_counts = np.unique(reference_raveled, return_counts=True)

    source_cumulative = np.cumsum(source_pixel_counts).astype(np.float)
    source_cumulative /= source_cumulative[-1]

    reference_cumulative = np.cumsum(reference_pixel_counts).astype(np.float)
    reference_cumulative /= reference_cumulative[-1]

    interpolated_image_pixel_values = np.interp(source_cumulative, reference_cumulative, reference_pixel_val_arr)

    return interpolated_image_pixel_values[reverse].reshape(shape)

sourceFileName = sys.argv[1]
referenceFileName = sys.argv[2]

source = mpimg.imread(sourceFileName)
reference = mpimg.imread(referenceFileName)

sourceR = np.zeros(source.shape)
sourceG = np.zeros(source.shape)
sourceB = np.zeros(source.shape)

referenceR = np.zeros(reference.shape)
referenceG = np.zeros(reference.shape)
referenceB = np.zeros(reference.shape)

sourceR, sourceG, sourceB = source[:,:,0], source[:,:,1], source[:,:,2]
referenceR, referenceG, referenceB = reference[:,:,0], reference[:,:,1], reference[:,:,2]

matchedR = histogram_matching(sourceR, referenceR)
matchedG = histogram_matching(sourceG, referenceG)
matchedB = histogram_matching(sourceB, referenceB)

matchedR /= 255
matchedG /= 255
matchedB /= 255

matched = histogram_matching(source,reference)
matched /= 255 


plt.imshow(matched)
plt.savefig(sourceFileName + "_histmatch.jpg")
plt.show()






        





    






