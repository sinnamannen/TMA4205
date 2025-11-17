from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

def calc_Ix(I):
    dx = np.zeros_like(I)
    dx[:, :-1] = I[:, 1:] - I[:, :-1]   #updating for i < m
    dx[:, -1] = I[:, -1] - I[:, -2]     #updating for i = m
    return dx

def calc_Iy(I):
    dy = np.zeros_like(I)
    dy[:-1, :] = I[1:, :] - I[:-1, :]   #updating for i < m
    dy[-1, :] = I[-1, :] - I[-2, :]     #updating for i = m
    return dy

def calc_derivatives(I0, I1):
    # x-derivative
    Ix0 = calc_Ix(I0)
    Ix1 = calc_Ix(I1)
    Ix = 0.5 * (Ix0 + Ix1)

    # y-derivative
    Iy0 = calc_Iy(I0)
    Iy1 = calc_Iy(I1)
    Iy = 0.5 * (Iy0 + Iy1)

    # t-derivative
    It = I1 - I0

    return Ix, Iy, It

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

def calc_rhs(Ix, Iy, It):
    rhsu = -It * Ix
    rhsv = -It * Iy
    return rhsu, rhsv

def load_image_2(filepath):
    img = Image.open(filepath).convert('L')  # Convert to grayscale
    img = np.array(img, dtype=np.float32)
    return img

def load_image(filepath):
    img = plt.imread(filepath)
    img = img.astype(np.float32)
    
    return img

def preprocess_image(img, sigma=1.0):
    # Apply Gaussian smoothing
    smoothed_img = gaussian_filter(img, sigma=sigma)
    return smoothed_img



def get_derivatives_and_rhs(img0, img1, from_file=False, sigma=0):
    print("Using per og henning preprocessing")
    if from_file:
        img0 = load_image(img0)
        img1 = load_image(img1)
    if sigma > 0:
        img0 = preprocess_image(img0, sigma=sigma)
        img1 = preprocess_image(img1, sigma=sigma)
    Ix, Iy, It = calc_derivatives(img0, img1)
    rhsu, rhsv = calc_rhs(Ix, Iy, It)
    return Ix, Iy, rhsu, rhsv


def get_rhs(Ix, Iy, It):
    rhsu = -It * Ix
    rhsv = -It * Iy
    return rhsu, rhsv


def get_derivatives_and_rhs_2(img0, img1, from_file=False, sigma=0):
    print("Using våres preprocessing")
    if from_file:
        img0 = load_image_2(img0)
        img1 = load_image_2(img1)
    if sigma > 0:
        img0 = preprocess_image(img0, sigma=sigma)
        img1 = preprocess_image(img1, sigma=sigma)
    Ix, Iy, It = calculate_image_derivatives(img0, img1)
    rhsu, rhsv = get_rhs(Ix, Iy, It)
    return Ix, Iy, rhsu, rhsv

Ix1, Iy1, rhsu1, rhsv1 = get_derivatives_and_rhs("frame10.png", "frame11.png", from_file=True, sigma=1.0)
Ix2, Iy2, rhsu2, rhsv2 = get_derivatives_and_rhs_2("frame10.png", "frame11.png", from_file=True, sigma=1.0)

print("Difference in Ix:", np.linalg.norm(Ix1 - Ix2))
print("Difference in Iy:", np.linalg.norm(Iy1 - Iy2))
print("Difference in rhsu:", np.linalg.norm(rhsu1 - rhsu2))
print("Difference in rhsv:", np.linalg.norm(rhsv1 - rhsv2))

print(Ix1.max())
print(Ix2.max())