# %% [markdown]
# # CS445: Computational Photography
# ## Programming Project 4: Image-Based Lighting
# 

# %% [markdown]
# ## Recovering HDR Radiance Maps 
# 
# Load libraries and data

# %%
# jupyter extension that allows reloading functions from imports without clearing kernel :D
%load_ext autoreload
%autoreload 2

# %%
# from google.colab import drive
# drive.mount('/content/drive')

# %%
# System imports
from os import path
import math

# Third-Party Imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

# modify to where you store your project data including utils
datadir = "/content/drive/My Drive/cs445_projects/proj4/" 

utilfn = datadir + "utils"
!cp -r "$utilfn" .
samplesfn = datadir + "samples"
!cp -r "$samplesfn" .

# can change this to your output directory of choice
!mkdir "images"
!mkdir "images/outputs"

# import starter code
import utils
from utils.io import read_image, write_image, read_hdr_image, write_hdr_image
from utils.display import display_images_linear_rescale, rescale_images_linear
from utils.hdr_helpers import gsolve
from utils.hdr_helpers import get_equirectangular_image
from utils.bilateral_filter import bilateral_filter



# %% [markdown]
# ### Reading LDR images
# 
# You can use the provided samples or your own images.  You get more points for using your own images, but it might help to get things working first with the provided samples.

# %%
# TODO: Replace this with your path and files

imdir = 'samples'
# imfns = ['0024.jpg', '0060.jpg', '0120.jpg', '0205.jpg', '0553.jpg']
# exposure_times = [1/24.0, 1/60.0, 1/120.0, 1/205.0, 1/553.0]

imfns =  ['19.jpeg', '20.jpeg', '21.jpeg','22.jpeg', '23.jpeg']
exposure_times =  [1/25.0, 1/60.0, 1/125.0,1/250.0, 1/501.0]

ldr_images = []
for f in np.arange(len(imfns)):
  im = read_image(imdir + '/' + imfns[f])
  dim = (403, 302)
  im = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
  if f==0:
    imsize = int((im.shape[0] + im.shape[1])/2) # set width/height of ball images
    ldr_images = np.zeros((len(imfns), imsize, imsize, 3))
  ldr_images[f] = cv2.resize(im, (imsize, imsize))

background_image_file = imdir + '/' + 'empty.jpg'
background_image = read_image(background_image_file)



# %% [markdown]
# ### Naive LDR merging 
# 
# Compute the HDR image as average of irradiance estimates from LDR images

# %%
def make_hdr_naive(ldr_images: np.ndarray, exposures: list) -> (np.ndarray, np.ndarray):
    '''
    Makes HDR image using multiple LDR images, and its corresponding exposure values.
    
    The steps to implement:
    1) Divide each image by its exposure time.
        - This will rescale images as if it has been exposed for 1 second.
    
    2) Return average of above images
    
    
    For further explanation, please refer to problem page for how to do it.
      
    Args:
        ldr_images(np.ndarray): N x H x W x 3  shaped numpy array representing
            N ldr images with width W, height H, and channel size of 3 (RGB)
        exposures(list): list of length N, representing exposures of each images.
            Each exposure should correspond to LDR images' exposure value.
    Return:
        (np.ndarray): H x W x 3 shaped numpy array representing HDR image merged using
            naive ldr merging implementation.
        (np.ndarray): N x H x W x 3  shaped numpy array represending log irradiances
            for each exposures
            
    '''
    N, H, W, C = ldr_images.shape
    # sanity check
    assert N == len(exposures)
  
    m_images = ldr_images.copy()

    hdr_image = np.zeros((H,W,C))


    for i in range(N):
        cur = m_images[i] / exposures[i]
        hdr_image += cur
        m = np.amin(cur)
        cur -= m 
        cur += 0.1
        m_images[i] = np.log(cur)
  
    return hdr_image/N, m_images



# %%
def display_hdr_image(im_hdr):
    '''
    Maps the HDR intensities into a 0 to 1 range and then displays. 
    Three suggestions to try: 
      (1) Take log and then linearly map to 0 to 1 range (see display.py for example) 
      (2) img_out = im_hdr / (1 + im_hdr)
      (3) HDR display code in a python package 
    '''


    m = np.amin(im_hdr)
    print(m)
    im_hdr -= m 
    im_hdr += 0.1
    le = np.log(im_hdr)


    le_min = le[le != -float('inf')].min()
    le_max = le[le != float('inf')].max()
    le[le==float('inf')] = le_max
    le[le==-float('inf')] = le_min

    le = (le - le_min) / (le_max - le_min)
    
    plt.imshow(le)
    


# %%

# get HDR image, log irradiance
naive_hdr_image, naive_log_irradiances = make_hdr_naive(ldr_images, exposure_times)

# write HDR image to directory
write_hdr_image(naive_hdr_image, 'images/outputs/naive_hdr.hdr')

# display HDR image
print('HDR Image')
display_hdr_image(naive_hdr_image)

# display original images (code provided in utils.display)
display_images_linear_rescale(ldr_images)

# display log irradiance image (code provided in utils.display)
display_images_linear_rescale(naive_log_irradiances)


# %% [markdown]
# ### Weighted LDR merging 
# 
# Compute HDR image as a weighted average of irradiance estimates from LDR images, where weight is based on pixel intensity so that very low/high intensities get less weight
# 

# %%
def make_hdr_weighted(ldr_images: np.ndarray, exposure_times: list) -> (np.ndarray, np.ndarray):
    '''
    Makes HDR image using multiple LDR images, and its corresponding exposure values.
    
    The steps to implement:
    1) compute weights for images with based on intensities for each exposures
        - This can be a binary mask to exclude low / high intensity values

    2) Divide each images by its exposure time.
        - This will rescale images as if it has been exposed for 1 second.
    
    3) Return weighted average of above images
    
    
    Args:
        ldr_images(np.ndarray): N x H x W x 3 shaped numpy array representing
            N ldr images with width W, height H, and channel size of 3 (RGB)
        exposure_times(list): list of length N, representing exposures of each images.
            Each exposure should correspond to LDR images' exposure value.
    Return:
        (np.ndarray): H x W x 3 shaped numpy array representing HDR image merged without
            under - over exposed regions

    '''
    N, H, W, C = ldr_images.shape
    # sanity check
    assert N == len(exposure_times)
    

    a = 128
    w = np.vectorize(lambda z: float(a-abs(z-a)))

    hdr_image = np.zeros((H,W,C))
    sum = np.zeros((H,W,C))

    for l in range(N):
        cur = ldr_images[l]/ exposure_times[l]
        weight = w(ldr_images[l]*255)
        sum += weight
        hdr_image += weight * cur
        
   
    return hdr_image / sum

# %%
# get HDR image, log irradiance
weighted_hdr_image = make_hdr_weighted(ldr_images, exposure_times)

# write HDR image to directory
write_hdr_image(weighted_hdr_image, 'images/outputs/weighted_hdr.hdr')

# display HDR image
display_hdr_image(weighted_hdr_image)


# %% [markdown]
# Display of difference between naive and weighted for your own inspection
# 
# Where does the weighting make a big difference increasing or decreasing the irradiance estimate?  Think about why.

# %%
# display difference between naive and weighted 

log_diff_im = np.log(weighted_hdr_image)-np.log(naive_hdr_image)
print('Min ratio = ', np.exp(log_diff_im).min(), '  Max ratio = ', np.exp(log_diff_im).max())
plt.figure()
plt.imshow(rescale_images_linear(log_diff_im))

# %% [markdown]
# ### LDR merging with camera response function estimation 
# 
# Compute HDR after calibrating the photometric reponses to obtain more accurate irradiance estimates from each image
# 
# Some suggestions on using <tt>gsolve</tt>:
# <ul>
# 	<li>When providing input to gsolve, don't use all available pixels, otherwise you will likely run out of memory / have very slow run times. To overcome, just randomly sample a set of pixels (1000 or so can suffice), but make sure all pixel locations are the same for each exposure.</li>
# 	<li>The weighting function w should be implemented using Eq. 4 from the paper (this is the same function that can be used for the previous LDR merging method).</li>
# 	<li>Try different lambda values for recovering <i>g</i>. Try lambda=1 initially, then solve for <i>g</i> and plot it. It should be smooth and continuously increasing. If lambda is too small, g will be bumpy.</li>
# 	<li>Refer to Eq. 6 in the paper for using g and combining all of your exposures into a final image. Note that this produces log irradiance values, so make sure to exponentiate the result and save irradiance in linear scale.</li>
# </ul>

# %%
def make_hdr_estimation(ldr_images: np.ndarray, exposure_times: list, lm)-> (np.ndarray, np.ndarray):
    '''
    Makes HDR image using multiple LDR images, and its corresponding exposure values.
    Please refer to problem notebook for how to do it.
    
    **IMPORTANT**
    The gsolve operations should be ran with:
        Z: int64 array of shape N x P, where N = number of images, P = number of pixels
        B: float32 array of shape N, log shutter times
        l: lambda; float to control amount of smoothing
        w: function that maps from float intensity to weight  
    The steps to implement:
    1) Create random points to sample (from mirror ball region)
    2) For each exposures, compute g values using samples
    3) Recover HDR image using g values
    

    Args:
        ldr_images(np.ndarray): N x H x W x 3 shaped numpy array representing
            N ldr images with width W, height H, and channel size of 3 (RGB)
        exposures(list): list of length N, representing exposures of each images.
            Each exposure should correspond to LDR images' exposure value.
        lm (scalar): the smoothing parameter
    Return:
        (np.ndarray): H x W x 3 shaped numpy array representing HDR image merged using
            gsolve
        (np.ndarray): N x H x W x 3 shaped numpy array represending log irradiances
            for each exposures
        (np.ndarray): 3 x 256 shaped numpy array represending g values of each pixel intensities
            at each channels (used for plotting)
    '''
    N, H, W, C = ldr_images.shape
    # sanity check
    assert N == len(exposure_times)
    
    # TO DO: implement HDR estimation using gsolve
    # gsolve(Z, B, l, w) -> g, lE

    hdr = np.zeros((H, W, 3))

    P = 1000
    pr = np.random.randint(H, size = P)
    pc = np.random.randint(W, size = P)

    B = np.log(exposure_times)
    k = 128 
    w = np.vectorize(lambda z: k - np.abs(z - k))


    ldr_images = ldr_images * 255.0

    c_log_i = np.zeros((N, H, W, 3))

    g_plot = np.zeros((3, 256))


    def channel_solve(B, lm, w, N, P, C):
        Z = np.zeros((N, P))
        for i in range(N):
            for j in range(P):
                Z[i, j] = ldr_images[i, pr[j], pc[j], C]
        g, lE = gsolve(Z.astype("uint8"), B, lm, w)
        return g




    for i in range(C):

        g = channel_solve(B, lm, w, N, P, i)
        g_plot[i] = g


        for j in range(H):
            for l in range(W):
                sum_i = 0
                sum_w = 0
                for n in range(N):
                    val = int(ldr_images[n, j, l, i])

                    cur = g[val] - B[n]

                    weight = w(val)
                    sum_w += weight
                    sum_i += weight * cur
                    c_log_i[n, j, l, i] = cur

                hdr[j, l, i] = sum_i / sum_w

    hdr = np.exp(hdr)

    return hdr, c_log_i, g_plot





# %%
lm = 10
# get HDR image, log irradiance
calib_hdr_image, calib_log_irradiances, g = make_hdr_estimation(ldr_images, exposure_times, lm)

# write HDR image to directory
write_hdr_image(calib_hdr_image, 'images/outputs/calib_hdr.hdr')

# display HDR image
display_hdr_image(calib_hdr_image)


# %% [markdown]
# The following code displays your results. You can copy the resulting images and plots directly into your report where appropriate.

# %%
# display difference between calibrated and weighted
log_diff_im = np.log(calib_hdr_image/calib_hdr_image.mean())-np.log(weighted_hdr_image/weighted_hdr_image.mean())
print('Min ratio = ', np.exp(log_diff_im).min(), '  Max ratio = ', np.exp(log_diff_im).max())
plt.figure()
plt.imshow(rescale_images_linear(log_diff_im))

# display original images (code provided in utils.display)
display_images_linear_rescale(ldr_images)

# display log irradiance image (code provided in utils.display)
display_images_linear_rescale(calib_log_irradiances)

# plot g vs intensity, and then plot intensity vs g
N, NG = g.shape
labels = ['R', 'G', 'B']
plt.figure()
for n in range(N):
    plt.plot(g[n], range(NG), label=labels[n])
plt.gca().legend(('R', 'G', 'B'))

plt.figure()
for n in range(N):
    plt.plot(range(NG), g[n], label=labels[n])
plt.gca().legend(('R', 'G', 'B'))

# %%
def weighted_log_error(ldr_images, hdr_image, log_irradiances):
  # computes weighted RMS error of log irradiances for each image compared to final log irradiance
  N, H, W, C = ldr_images.shape
  w = 1-abs(ldr_images - 0.5)*2
  err = 0
  for n in np.arange(N):
    err += np.sqrt(np.multiply(w[n], (log_irradiances[n]-np.log(hdr_image))**2).sum()/w[n].sum())/N 
  return err


# compare solutions
err = weighted_log_error(ldr_images, naive_hdr_image, naive_log_irradiances)
print('naive:  \tlog range = ', round(np.log(naive_hdr_image).max() - np.log(naive_hdr_image).min(),3), '\tavg RMS error = ', round(err,3))
err = weighted_log_error(ldr_images, weighted_hdr_image, naive_log_irradiances)
print('weighted:\tlog range = ', round(np.log(weighted_hdr_image).max() - np.log(weighted_hdr_image).min(),3), '\tavg RMS error = ', round(err,3))
err = weighted_log_error(ldr_images, calib_hdr_image, calib_log_irradiances)
print('calibrated:\tlog range = ', round(np.log(calib_hdr_image).max() - np.log(calib_hdr_image).min(),3), '\tavg RMS error = ', round(err,3))

# display log hdr images (code provided in utils.display)
display_images_linear_rescale(np.log(np.stack((naive_hdr_image/naive_hdr_image.mean(), weighted_hdr_image/weighted_hdr_image.mean(), calib_hdr_image/calib_hdr_image.mean()), axis=0)))

# %% [markdown]
# ## Panoramic transformations 
# 
# Compute the equirectangular image from the mirrorball image

# %%
def panoramic_transform(hdr_image):
    '''
    Given HDR mirror ball image, 
    
    Expects mirror ball image to have center of the ball at center of the image, and
    width and height of the image to be equal.
    
    Steps to implement:
    1) Compute N image of normal vectors of mirror ball
    2) Compute R image of reflection vectors of mirror ball
    3) Map reflection vectors into spherical coordinates
    4) Interpolate spherical coordinate values into equirectangular grid.
    
    Steps 3 and 4 are implemented for you with get_equirectangular_image

    '''
    H, W, C = hdr_image.shape
    assert H == W
    assert C == 3

    half = H // 2
    hdr_image = hdr_image * 255.0
    R = np.zeros((H, W, 3))
    N = np.zeros((H, W, 3))

    for i in range(W):
        for j in range(H):
            nx = (i - half) / half
            ny = (j - half) / half

            val = 1 - np.square(nx) - np.square(ny)
            if val < 0:
                nx = 0
                ny = 0
                nz = 0
            else:
                nz =  math.sqrt(val)

            N[j][i][:] = np.array([nx, ny, nz])
            
            V = np.array([0, 0, -1])
            R[j][i][:] = (V - 2 * np.dot(V, N[j][i]) * N[j][i])
    
        

    # TO DO: compute N and R
        
    # R = V - 2 * dot(V,N) * N


    
    plt.imshow((N+1)/2)
    plt.show()
    plt.imshow((R+1)/2)
    plt.show()

    equirectangular_image = get_equirectangular_image(R, hdr_image)
    return equirectangular_image

# %%
hdr_mirrorball_image = read_hdr_image('images/outputs/calib_hdr.hdr')
eq_image = panoramic_transform(hdr_mirrorball_image)

write_hdr_image(eq_image, 'images/outputs/equirectangular.hdr')

plt.figure(figsize=(15,15))
display_hdr_image(eq_image)


# %% [markdown]
# ---

# %% [markdown]
# ## Rendering synthetic objects into photographs 
# 
# Use Blender to render the scene with and with objects and obtain the mask image.  The code below should then load the images and create the final composite.

# %%
# Read the images that you produced using Blender.  Modify names as needed.
O = read_image('images/proj4_objects.png')
E = read_image('images/proj4_empty.png')
M = read_image('images/proj4_mask.png')
# plt.imshow(M)
# print(M[250][200])

M = M > 0.5
I = background_image
I = cv2.resize(I, (M.shape[1], M.shape[0]))

# %%
# TO DO: compute final composite
c = 1
R = O
result = M*R + (1-M)*I + (1-M)*(R-E)*c

plt.figure(figsize=(20,20))
plt.imshow(result)
plt.show()

write_image(result, 'images/outputs/final_composite.png')

# %% [markdown]
# ---

# %% [markdown]
# 
# ## Bells & Whistles (Extra Points)
# 
# ### Additional Image-Based Lighting Result 
# 
# 
# ### Other panoramic transformations 
# 
# 
# ### Photographer/tripod removal 
# 
# 
# ### Local tonemapping operator 
# 
# 


