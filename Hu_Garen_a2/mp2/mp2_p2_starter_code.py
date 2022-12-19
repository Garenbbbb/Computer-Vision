# Libraries you will find useful
import numpy as np
import scipy 
import skimage
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage.filters import gaussian_laplace

# Starter code for Assignment 2 Part 2: Scale-space blob detection



# convert images to grayscale
# rescale the intensities to between 0 and 1 (simply divide them by 255 should do the trick)
img_file = 'part2_images/butterfly.jpg'
img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
img /= 255.0
shape = img.shape


# Create the Laplacian filter
# Pay careful attention to setting the right filter mask size. Hint: Should the filter width be odd or even?



k = 2
factor = 1.25

iteration = 10

scale_space = np.empty((shape[0], shape[1], iteration))

for i in range(iteration):
    gol_img = gaussian_laplace(img, k)
    scale_space[:, :, i] = gol_img**2
    k *= factor


# filtering the image (two implementations)
# one that increases filter size, and one that downsamples the image
# For timing, use time.time()



# nonmaximum suppression in scale space
# you may find functions scipy.ndimage.filters.rank_filter or scipy.ndimage.filters.generic_filter useful



# To display the detected regions as circle
from matplotlib.patches import Circle
def show_all_circles(image, cx, cy, rad, color='r'):
    """
    image: numpy array, representing the grayscsale image
    cx, cy: numpy arrays or lists, centers of the detected blobs
    rad: numpy array or list, radius of the detected blobs
    """
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(image, cmap='gray')
    for x, y, r in zip(cx, cy, rad):
        circ = Circle((x, y), r, color=color, fill=False)
        ax.add_patch(circ)

    plt.title('%i circles' % len(cx))
    plt.show()

