from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter

def load_image(filepath):
    img = Image.open(filepath).convert('L')  # Convert to grayscale
    img = np.array(img, dtype=np.float32)
    
    return img

def preprocess_image(img, sigma=1.0):
    # Apply Gaussian smoothing
    smoothed_img = gaussian_filter(img, sigma=sigma)
    return smoothed_img

def calculate_image_derivatives(img0, img1):
    It = img1 - img0
    #Note flipped indexes in x and y direction, because of matrix...
    #Calculate Ix
    Ix_0 = np.zeros_like(img0) 
    Ix_1 = np.zeros_like(img0)

    Ix_0[:,:-1] = img0[:, 1:] - img0[:, :-1]
    Ix_0[:,-1] = Ix_0[:,-2]

    Ix_1[:,:-1] = img1[:, 1:] - img1[:, :-1]
    Ix_1[:,-1] = Ix_1[:,-2]

    Ix = (Ix_0 + Ix_1) / 2.0
    #Calculate Iy
    Iy_0 = np.zeros_like(img0) 
    Iy_1 = np.zeros_like(img0)
    
    Iy_0[:-1,:] = img0[1:,:] - img0[:-1,:]
    Iy_0[-1,:] = Iy_0[-2,:]

    Iy_1[:-1,:] = img1[1:,:] - img1[:-1,:]
    Iy_1[-1,:] = Iy_1[-2,:]

    Iy = (Iy_0 + Iy_1) / 2.0
    
    return Ix, Iy, It

def get_derivatives_and_rhs(img0, img1, from_file=False, sigma=0):
    if from_file:
        img0 = load_image(img0)
        img1 = load_image(img1)
    if sigma > 0:
        img0 = preprocess_image(img0, sigma=sigma)
        img1 = preprocess_image(img1, sigma=sigma)
    Ix, Iy, It = calculate_image_derivatives(img0, img1)
    rhsu, rhsv = get_rhs(Ix, Iy, It)
    return Ix, Iy, rhsu, rhsv



def get_rhs(Ix, Iy, It):
    rhsu = -It * Ix
    rhsv = -It * Iy
    return rhsu, rhsv

