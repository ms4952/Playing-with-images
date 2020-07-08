import pickle

import numpy as np
import argparse
import imageio
import logging
import sys
from scipy.ndimage.filters import convolve

# Displacements are by default saved to a file after every run. Once you have confirmed your
# LK code is working, you can load saved displacements to save time testing the
# rest of the project.
DEFAULT_DISPLACEMENTS_FILE = "final_displacements.pkl"

def bilinear_interp(image, points):
    """Given an image and an array of row/col (Y/X) points, perform bilinear
    interpolation and return the pixel values in the image at those points."""
    points = np.asarray(points)
    if points.ndim == 1:
        points = points[np.newaxis]

    valid = np.all(points < [image.shape[0]-1, image.shape[1]-1], axis=-1)
    valid *= np.all(points >= 0, axis=-1)
    valid = valid.astype(np.float32)
    points = np.minimum(points, [image.shape[0]-2, image.shape[1]-2])
    points = np.maximum(points, 0)

    fpart, ipart = np.modf(points)
    tl = ipart.astype(np.int32)
    br = tl+1
    tr = np.concatenate([tl[..., 0:1], br[..., 1:2]], axis=-1)
    bl = np.concatenate([br[..., 0:1], tl[..., 1:2]], axis=-1)

    b = fpart[..., 0:1]
    a = fpart[..., 1:2]

    top = (1-a) * image[tl[..., 0], tl[..., 1]] + \
        a * image[tr[..., 0], tr[..., 1]]
    bot = (1-a) * image[bl[..., 0], bl[..., 1]] + \
        a * image[br[..., 0], br[..., 1]]
    return ((1-b) * top + b * bot) * valid[..., np.newaxis]


def translate(image, displacement):
    """Takes an image and a displacement of the form X,Y and translates the
    image by the displacement. The shape of the output is the same as the
    input, with missing pixels filled in with zeros."""
    pts = np.mgrid[:image.shape[0], :image.shape[1]
                   ].transpose(1, 2, 0).astype(np.float32)
    pts -= displacement[::-1]

    return bilinear_interp(image, pts)


def convolve_img(image, kernel):
    """Convolves an image with a convolution kernel. Kernel should either have
    the same number of dimensions and channels (last dimension shape) as the
    image, or should have 1 less dimension than the image."""
    if kernel.ndim == image.ndim:
        if image.shape[-1] == kernel.shape[-1]:
            return np.dstack([convolve(image[..., c], kernel[..., c]) for c in range(kernel.shape[-1])])
        elif image.ndim == 2:
            return convolve(image, kernel)
        else:
            raise RuntimeError("Invalid kernel shape. Kernel: %s Image: %s" % (
                kernel.shape, image.shape))
    elif kernel.ndim == image.ndim - 1:
        return np.dstack([convolve(image[..., c], kernel) for c in range(image.shape[-1])])
    else:
        raise RuntimeError("Invalid kernel shape. Kernel: %s Image: %s" % (
            kernel.shape, image.shape))


def gaussian_kernel(ksize=5):
    """
    Computes a 2-d gaussian kernel of size ksize and returns it.
    """
    kernel = np.exp(-np.linspace(-(ksize//2), ksize//2, ksize)
                    ** 2 / 2) / np.sqrt(2*np.pi)
    kernel = np.outer(kernel, kernel)
    kernel /= kernel.sum()
    return kernel


def lucas_kanade(H, I):
    """Given images H and I, compute the displacement that should be applied to
    H so that it aligns with I."""


    # motion in dark regions is difficult to estimate. Generate a binary mask
    # indicating pixels that are valid (average color value > 0.25) in both H
    # and I.
    mask = (H.mean(-1) > 0.25) * (I.mean(-1) > 0.25)
    mask = mask[:, :, np.newaxis]
    sobelKernel_x=([-1,0,1],[-2,0,2],[-1,0,1])
    sobelKernel_y=([1,2,1],[0,0,0],[-1,-2,-1])
    sobelKernel_x=np.array(sobelKernel_x)
    sobelKernel_x=-1*sobelKernel_x/8
    sobelKernel_y=np.array(sobelKernel_y)/8
    
    I_x=convolve_img(I,sobelKernel_x)
    I_y=convolve_img(I,sobelKernel_y)
    I_t=I-H
    
    Ixx=(I_x*I_x)*mask
    Ixy = (I_x*I_y)*mask
    Ixt = (I_x*I_t)*mask
    
    Iyy = (I_y*I_y)*mask            
    Iyt = (I_y*I_t)*mask
    AtA=np.array([[np.sum(Ixx),np.sum(Ixy)],[np.sum(Ixy),np.sum(Iyy)]])
   
    Atb=np.array([[-1*np.sum(Ixt)],[-1*np.sum(Iyt)]]).reshape(-1)
    
    #displacement=np.linalg.solve(AtA,Atb)
    displacement=np.dot(np.linalg.pinv(AtA),Atb)
    # Compute the partial image derivatives w.r.t. X, Y, and Time (t).
    # In other words, compute I_y, I_x, and I_t
    # To achieve this, use a _normalized_ 3x3 sobel kernel and the convolve_img
    # function above. NOTE: since you're convolving the kernel, you need to 
    # multiply it by -1 to get the proper direction.

    # Compute the various products (Ixx, Ixy, Iyy, Ixt, Iyt) necessary to form
    # AtA. Apply the mask to each product.

    # Build the AtA matrix and Atb vector. You can use the .sum() function on numpy arrays to help.

    # Solve for the displacement using linalg.solve

    # return the displacement and some intermediate data for unit testing..
    return displacement, AtA, Atb


def iterative_lucas_kanade(H, I, steps):
    # Run the basic Lucas Kanade algorithm in a loop `steps` times.
    # Start with an initial displacement of 0 and accumulate displacements.
    disp = np.zeros((2,), np.float32)
    for i in range(steps):
        Ht=translate(H, disp)
        # run Lucas Kanade and update the displacement estimate
        
        x=lucas_kanade(Ht, I)
        disp=np.add(disp,x[0])
    return disp


def gaussian_pyramid(image, levels):
    """
    Builds a Gaussian pyramid for an image with the given number of levels, then return it.
    Inputs:
        image: a numpy array (i.e., image) to make the pyramid from
        levels: how many levels to make in the gaussian pyramid
    Retuns:
        An array of images where each image is a blurred and shruken version of the first.
    """
    kernel=gaussian_kernel()
    # Compute a gaussian kernel using the gaussian_kernel function above. You can leave the size as default.

    # Add image to the the list as the first level
    pyr = [image]
    for level in range(1, levels):
        c=convolve_img(pyr[level-1], kernel)
        
        c= c[0::2,0::2]
        # Convolve the previous image with the gussian kernel
        
        # decimate the convolved image by downsampling the pixels in both dimensions.
        # Note: you can use numpy advanced indexing for this (i.e., ::2)
        
        # add the sampled image to the list
        pyr.append(c)
        
    return pyr


def pyramid_lucas_kanade(H, I, initial_d, levels, steps):
    """Given images H and I, and an initial displacement that roughly aligns H
    to I when applied to H, run Iterative Lucas Kanade on a pyramid of the
    images with the given number of levels to compute the refined
    displacement."""

    initial_d = np.asarray(initial_d, dtype=np.float32)

    # Build Gaussian pyramids for the two images.
    
    H1=gaussian_pyramid(H,levels)
    I1=gaussian_pyramid(I,levels)
    # Start with an initial displacement (scaled to the coarsest level of the
    # pyramid) and compute the updated displacement at each level using Lucas
    # Kanade.
    disp = initial_d / 2.**(levels)
    for level in range(levels):
        # Get the two images for this pyramid level.
        H2=H1[len(H1)-level-1]
        I2=I1[len(H1)-level-1]
        
        
        # Get the two images for this pyramid level.
        disp=disp*2
        H3=translate(H2,disp)
        #H1=np.array(H1)
        #I1=np.array(I1)
        
        ilk=iterative_lucas_kanade(H3,I2,steps)
        
        #ilk=np.around(ilk, decimals=2)
        
        disp=np.add(disp,ilk)
        #disp=np.around(disp,1)
        
        # Scale the previous level's displacement and apply it to one of the
        # between the two images at this level.
        
        # Use the iterative Lucas Kanade method to compute a displacement
        # between the two images at this level.

        # Update the displacement based on the one you just computed.
        
    print("new disp",disp)

        # Scale the previous level's displacement and apply it to one of the
        # images via translation.
        

        # Use the iterative Lucas Kanade method to compute a displacement
        # between the two images at this level.
        

        # Update the displacement based on the one you just computed.
        

    # Return the final displacement.
    return disp


def track_object(frame1, frame2, boundingBox, steps):
    """
    Attempts to track the object defined by window from frame one to
    frame two.

    args:
        frame1 - the first frame in the sequence
        frame2 - the second frame in the sequence
        boundingBox - A bounding box (x, y, w, h) around the object in the first frame
    """
    # get the x,y,w,h from the bounding box
    
    x, y, w, h =boundingBox
    # extract a sub-image from each frame using the bounding box
    # use the first frame for H and the second for I
    H = frame1[y:y+h,x:x+w]
    I = frame2[y:y+h,x:x+w]
    levels =int(np.log2(min(w,h)))
    #print("levels",levels)
    initial_displacement = np.array([0, 0])
    flow =pyramid_lucas_kanade(H, I,initial_displacement, levels, steps)

    return flow


def visualize(first, firstBB, second, secondBB):
    import matplotlib
    import matplotlib.pyplot as plt

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    # Show the source image
    ax1.imshow(first)
    ax1.set_title('First Frame')
    rect = matplotlib.patches.Rectangle(
        (firstBB[0], firstBB[1]), firstBB[2], firstBB[3], edgecolor='r', facecolor="none")
    ax1.add_patch(rect)

    # Show the second
    ax2.imshow(second)
    ax2.set_title('Second Frame')
    rect = matplotlib.patches.Rectangle(
        (secondBB[0], secondBB[1]), secondBB[2], secondBB[3], edgecolor='r', facecolor="none")
    ax2.add_patch(rect)

    plt.show()


if __name__ == "__main__":
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Tracks an object between two images by computing the optical flow using lucas kanade.')
    parser.add_argument(
        'firstFrame', type=str, help='The first image in the sequence.')
    parser.add_argument(
        'secondFrame', type=str, help='The second image in the sequence.')
    parser.add_argument('--boundingBox', type=str,
                        help='The bounding box of the object in x,y,w,h format', default='304,329,106,58')
    parser.add_argument('--steps', type=int,
                        help='The number of steps to use', default=5)
    parser.add_argument('--visualize',
                        help='Visualize the results', action='store_true')
    args = parser.parse_args()

    # load the images and parse out the bounding box.
    boundingBox = np.array([int(x) for x in args.boundingBox.split(',')])
    first = imageio.imread(args.firstFrame)[
        :, :, :3].astype(np.float32) / 255.0
    second = imageio.imread(args.secondFrame)[
        :, :, :3].astype(np.float32) / 255.0
    flow = track_object(first, second, boundingBox, args.steps)
    
    # Use the flow to move the bouding box.
    resultBoundingBox = boundingBox.copy().astype(np.float32)
    resultBoundingBox[0:2] += flow

    print('tracked object to have moved %s to %s' % (str(flow), str((resultBoundingBox[0], resultBoundingBox[1]))))

    if args.visualize:
        visualize(first, boundingBox, second, resultBoundingBox)
