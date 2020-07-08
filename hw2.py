
"""
Imports we need.
Note: You may _NOT_ add any more imports than these.
"""
import argparse
import imageio
import logging
import numpy as np
from PIL import Image


def load_image(filename):
    im = Image.open(filename)
    return np.array(im)


def build_A(pts1, pts2):
    
    if pts1.shape != pts2.shape:
        raise ValueError('The source points for homography computation must have the same shape (%s vs %s)' % (
            str(pts1.shape), str(pts2.shape)))
    if pts1.shape[0] < 4:
        raise ValueError('There must be at least 4 pairs of correspondences.')
    num_pts = pts1.shape[0]
    A =np.zeros(shape=(2*pts1.shape[0],9))
    m=0
    n=0
    while m!=num_pts:
        x = pts1[m][0]
        x1=pts2[m][0]
        y=pts1[m][1]
        y1=pts2[m][1]

        A[n]=[x, y, 1, 0, 0, 0, -x1*x, -x1*y, -x1 ]
        A[n+1]=[0, 0, 0, x, y, 1, -y1*x, -y1*y, -y1 ]
        n+=2
        m+=1
    

    return A


def compute_H(pts1, pts2):
    
    # TODO: Construct the intermediate A matrix using build_A
    A = build_A(pts1, pts2)

    # TODO: Compute the symmetric matrix AtA.
    
    AtA=np.dot(A.transpose(),A)

    # TODO: Compute the eigenvalues and eigenvectors of AtA.
    eig_vals, eig_vecs = np.linalg.eigh(AtA)

    # TODO: Determine which eigenvalue is the smallest
    
    min_eig_val_index = np.where(eig_vals== np.amin(eig_vals))

    
    min_eig_vec = eig_vecs[:,min_eig_val_index].reshape(3,3)

    return min_eig_vec


def bilinear_interp(image, point):
    
    x=point[0]
    y=point[1]

    # TODO: Compute i,j as the integer parts of x, y

    i=int(x)
    j=int(y)

    # TODO: check that i + 1 and j + 1 are within range of the image. if not, just return the pixel at i, j
    if j+1>=image.shape[0] or i+1>=image.shape[1]:
        return image[j][i]
    else:
        


    # TODO: Compute a and b as the floating point parts of x, y

        a = x%1
        b = y%1

        

        f=(1-a)*(1-b)*image[j][i]+a*(1-b)*image[j][i+1]+a*b*image[j+1][i+1]+(1-a)*b*image[j+1][i]
        return f



def apply_homography(H, points):
    
     
    homogenous_points=np.concatenate((points,np.ones(shape=(points.shape[0],1))), axis=1)

    # TODO: Apply the homography

    result=[]
    
    for r in range(len(homogenous_points)):   
        R=np.dot(H,homogenous_points[r])
        #print(R)
        z=R[len(R)-1]
        x=R[0]
        y=R[1]
        result.append([x/z,y/z])
        
        

    result= np.array(result)
    return result

    # TODO: Apply the homography

    # TODO: Convert the result back to cartesian coordinates and return the results


def warp_homography(source, target_shape, Hinv):
    
    # TODO: allocation a numpy array of zeros that is size target_shape and the same type as source.
    result = np.zeros(target_shape, dtype='uint8')
    
    height=target_shape[0]
    width=target_shape[1]
    channels=target_shape[2]
    


    
    for x in range(width):
        for y in range(height):
            # TODO: apply the homography to the x,y location
           
            points = [x,y]
            points=np.array(points).reshape((1,2))
            ah = apply_homography(Hinv,points)
            m=ah[0][0]
            n=ah[0][1]
            #i, j = h_result
            if m > 0 and n > 0:
                if n<=source.shape[0] and m<=source.shape[1]:
                    result[y][x] = bilinear_interp(source, ah[0])
                    
                

    return result



def rectify_image(image, source_points, target_points, crop):
    
    H = compute_H(source_points, target_points)

    
    
    w=image.shape[1]
    h=image.shape[0]
    topleft=[0,0]
    topright=[w,0]
    bottomleft=[0,h]
    bottomright=[w,h]
    boundingBox=[topleft,topright,bottomleft,bottomright]
    boundingBox=np.array(boundingBox)

    wbb = apply_homography(H,boundingBox)
    

    
    
    if crop:
        listx = sorted(wbb[:,0])
        listy = sorted(wbb[:,1])
        min_x = listx[1]
        min_y = listy[1]
    else:
        # TODO: Compute the min x and min y of the warped bounding box

        min_x = min(wbb[:,0])
        min_y = min(wbb[:,1])

        


    # TODO: Compute a translation matrix T such that min_x and min_y will go to zero
    T=([1,0,-min_x],[0,1,-min_y],[0,0,1])
    T = np.array(T)

    # TODO: Compute the rectified bounding box by applying the translation matrix to
    # the warped bounding box.

    rectified = apply_homography(T,wbb)


    inverseH = compute_H(rectified, boundingBox)


    # Determine the shape of the output image
    
    rx=sorted(rectified[:,0],reverse=True)
    ry=sorted(rectified[:,1],reverse=True)


    if crop:
        # TODO: Determine the side of the final output image as the second highest X and Y values of the
        # rectified bounding box
        foix=rx[1]
        foiy=ry[1]
    else:
        
        foix=rx[0]
        foiy=ry[0]
    
    
    shape = (int(foiy),int(foix),3)
    # TODO: Finally call warp_homography to rectify the image and return the result
    rectified_image = warp_homography(image, shape, inverseH)
    
    return rectified_image


def blend_with_mask(source, target, mask):
    m=mask.astype(np.float32) / np.max(mask)
    
    
    Result = (1-m)*target+m*source
    #Result = (m)*target+(1-m)*source
    #Result=np.uint8(Result)     
    

    return Result

def composite_image(source, target, source_pts, target_pts, mask):
    H=compute_H(target_pts,source_pts)

    # TODO: Warp the source image to a new image (that has the same shape as target) using the homography.
    

    source=warp_homography(source,(target.shape[0],target.shape[1],3), H)

    # TODO: Blend the warped images and return them.
    result=blend_with_mask(source,target,mask)

    return result



def rectify(args):
    
    source_points = np.array(args.source).reshape(4, 2)

    # load the destination points, or select some smart default ones if None
    if args.dst == None:
        height = np.abs(
            np.max(source_points[:, 1]) - np.min(source_points[:, 1]))
        width = np.abs(
            np.max(source_points[:, 0]) - np.min(source_points[:, 0]))
        args.dst = [0.0, height, 0.0, 0.0, width, 0.0, width, height]

    target_points = np.array(args.dst).reshape(4, 2)

    # load the input image
    logging.info('Loading input image %s' % (args.input))
    inputImage = load_image(args.input)

    # Compute the rectified image
    result = rectify_image(inputImage, source_points, target_points, args.crop)
    # save the result
    logging.info('Saving result to %s' % (args.output))
    imageio.imwrite(args.output, result)


def composite(args):
    """
    The 'main' function for the composite command.
    """

    # load the input image
    logging.info('Loading input image %s' % (args.input))
    inputImage = load_image(args.input)

    # load the target image
    logging.info('Loading target image %s' % (args.target))
    targetImage = load_image(args.target)

    # load the mask image
    logging.info('Loading mask image %s' % (args.mask))
    maskImage = load_image(args.mask)

    # If None, set the source points or sets them to the whole input image
    if args.source == None:
        (height, width, _) = inputImage.shape
        args.source = [0.0, height, 0.0, 0.0, width, 0.0, width, height]

    # Loads the source points into a 4-by-2 array
    source_points = np.array(args.source).reshape(4, 2)

    # Loads the target points into a 4-by-2 array
    target_points = np.array(args.dst).reshape(4, 2)

    # Compute the composite image
    result = composite_image(inputImage, targetImage,
                             source_points, target_points, maskImage)
    result=np.uint8(result)
    # save the result
    logging.info('Saving result to %s' % (args.output))
    imageio.imwrite(args.output, result)


"""
The main function
"""
if __name__ == '__main__':
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)
    logging.basicConfig(
        format='%(levelname)s: %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Warps an image by the computed homography between two rectangles.')
    subparsers = parser.add_subparsers(help='sub-command help')

    parser_rectify = subparsers.add_parser(
        'rectify', help='Rectifies an image such that the input rectangle is front-parallel.')
    parser_rectify.add_argument('input', type=str, help='The image to warp.')
    parser_rectify.add_argument('source', metavar='f', type=float, nargs=8,
                                help='A floating point value part of x1 y1 ... x4 y4')
    parser_rectify.add_argument(
        '--crop', help='If true, the result image is cropped.', action='store_true', default=False)
    parser_rectify.add_argument('--dst', metavar='x', type=float, nargs='+',
                                default=None, help='The four destination points in the output image.')
    parser_rectify.add_argument(
        'output', type=str, help='Where to save the result.')
    parser_rectify.set_defaults(func=rectify)

    parser_composite = subparsers.add_parser(
        'composite', help='Warps the input image onto the target points of the target image.')
    parser_composite.add_argument(
        'input', type=str, help='The source image to warp.')
    parser_composite.add_argument(
        'target', type=str, help='The target image to warp to.')
    parser_composite.add_argument('dst', metavar='f', type=float, nargs=8,
                                  help='A floating point value part of x1 y1 ... x4 y4 defining the box on the target image.')
    parser_composite.add_argument(
        'mask', type=str, help='A mask image the same size as the target image.')
    parser_composite.add_argument('--source', metavar='x', type=float, nargs='+',
                                  default=None, help='The four source points in the input image. If ommited, the whole image is used.')
    parser_composite.add_argument(
        'output', type=str, help='Where to save the result.')
    parser_composite.set_defaults(func=composite)

    args = parser.parse_args()
    args.func(args)
