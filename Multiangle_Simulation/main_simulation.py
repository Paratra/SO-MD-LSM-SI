
from biobeam import Bpm3d
import numpy as np
from pdb import set_trace as st
import matplotlib.pyplot as plt
from mayavi import mlab
from scipy.ndimage import convolve
from utils import generate_beads_array, apply_otf, gaussian_otf_3d_anisotropic, rotate_wavefront
import tifffile as tf
from scipy.spatial.transform import Rotation
from time import time
from tqdm import tqdm



def rotate_3d_array(array, angle, axis):
    """
    Rotate a 3D array by a given angle around a specified axis.

    Parameters:
    array (np.ndarray): The 3D array to rotate.
    angle (float): The rotation angle in degrees.
    axis (str): The axis to rotate around ('x', 'y', or 'z').

    Returns:
    np.ndarray: The rotated 3D array.
    """
    if axis not in ['x', 'y', 'z']:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    
    # Create the rotation matrix
    r = Rotation.from_euler(axis, angle, degrees=True)
    rotation_matrix = r.as_matrix()

    # Get the shape of the array
    shape = array.shape

    # Generate a grid of coordinates
    coords = np.indices(shape).reshape(3, -1)

    # Center the coordinates
    center = np.array(shape)[:, None] / 2
    centered_coords = coords - center

    # Rotate the coordinates
    rotated_coords = rotation_matrix @ centered_coords

    # Uncenter the coordinates
    rotated_coords += center

    # Round and clip the coordinates to valid indices
    rotated_coords = np.round(rotated_coords).astype(int)
    rotated_coords = np.clip(rotated_coords, 0, np.array(shape)[:, None] - 1)

    # Map the original array to the new coordinates
    rotated_array = np.zeros_like(array)
    for i in range(rotated_coords.shape[1]):
        rotated_array[tuple(rotated_coords[:, i])] = array[tuple(coords[:, i])]

    return rotated_array

def in_ellipsoid(x, y, z, a, b, c):
    """Check if a point (x, y, z) is inside the ellipsoid defined by semi-axes (a, b, c)."""
    return (x/a)**2 + (y/b)**2 + (z/c)**2 <= 1

def draw_sphere(grid, grid_seed, center, radius, grid_size, r_index):
    """Draw a sphere in a 3D grid at specified center with given radius."""
    x0, y0, z0 = center
    for x in range(max(0, x0-radius), min(grid_size, x0+radius+1)):
        for y in range(max(0, y0-radius), min(grid_size, y0+radius+1)):
            for z in range(max(0, z0-radius), min(grid_size, z0+radius+1)):
                if (x-x0)**2 + (y-y0)**2 + (z-z0)**2 <= radius**2:
                    grid[x, y, z] = r_index
                    grid_seed[x, y, z] = 1
# Parameters
num_spheres = 1300  # Number of spheres to generate
a, b, c = 90, 90, 90  # Semi-axes of the ellipsoid within the 512 grid
radius = 3  # Radius of spheres in pixels
grid_size = 256  # Size of the ndarray
RI = 1.1

# Initialize the grid
grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
grid_seed = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)

# Generate sphere centers randomly inside the ellipsoid
np.random.seed(42)  # for reproducibility
for ind in range(num_spheres):
    while True:
        # print(ind)
        # Generate random center within the grid
        x = np.random.randint(0, grid_size)
        y = np.random.randint(0, grid_size)
        z = np.random.randint(0, grid_size)
        # Adjust centers to be relative to the center of the grid
        if in_ellipsoid(x - grid_size//2, y - grid_size//2, z - grid_size//2, a, b, c):
            draw_sphere(grid, grid_seed,(x, y, z), radius, grid_size, r_index=RI)
            break





# create the refractive index difference
# N = 512
N = grid_size
dx = 0.135
r = 4
x = dx*(np.arange(N)-N//2)
Z, Y, X = np.meshgrid(x,x,x,indexing = "ij")
# R = np.sqrt(X**2+(Y-.9*r)**2+Z**2)
R = np.sqrt(X**2+(Y)**2+Z**2)
dn = 0.05*(R<r)


# get otf
na = 0.8
wavelength = 0.520  # micrometers
refractive_index = RI
grid_size = (N,N,N)  # 3D grid, should be odd dimensions
pixel_size_xy = 0.135  # micrometers per pixel in xy
pixel_size_z = 0.135   # micrometers per pixel in z


# Example usage
# beads_array, seed_array, beads_coors = generate_beads_array(3000, (512, 512, 512), 1.59)
beads_array = grid
seed_array = grid_seed


t_start = time()

rotate_angle = 270 # 60, 120, 270?
beads_array = rotate_3d_array(beads_array, angle=rotate_angle, axis='z')
seed_array = rotate_3d_array(seed_array, angle=rotate_angle, axis='z')

padding = [(0, 0), (0, 0), (N//2, N//2)]

# Apply padding
beads_array_expend = np.pad(beads_array, pad_width=padding, mode='constant', constant_values=0)
seed_array_expend = np.pad(seed_array, pad_width=padding, mode='constant', constant_values=0)

final_result_stack = np.zeros((N,N,N),dtype=np.float32)
for ind in tqdm(range(N)):

    beads_array = beads_array_expend[:,:,ind:ind+N]
    seed_array = seed_array_expend[:,:,ind:ind+N]
    # tf.imshow(np.swapaxes(beads_array,0,2))
    # plt.show()
    # st()
    # create the computational geometry
    # m = Bpm3d(dn = dn, units = (dx,)*3, lam = 0.488)
    m = Bpm3d(dn = beads_array, units = (dx,)*3, lam = 0.488, n0=RI)

    # # Adjust these parameters based on your needs
    # angle = 45  # Example: 45 degrees rotation
    # axis = 'z'  # Rotate around the x-axis
    # # Generate the initial cylindrical wavefront
    # u0 = m.u0_cylindrical(center=(0, 0), zfoc=None, NA=0.15)
    # # Rotate the wavefront to adjust the orientation
    # u0_rotated = rotate_wavefront(u0, angle, axis)
    # Propagate the rotated wavefront
    # u = m._propagate(u0=u0_rotated, return_comp="intens")


    # propagate a plane wave and return the intensity
    u = m.propagate(u0 = m.u0_cylindrical(NA = .2).T, return_comp = "intens")

    t_end = time()
    # print(t_end-t_start)



    otf = gaussian_otf_3d_anisotropic(na, wavelength, refractive_index, grid_size, pixel_size_xy, pixel_size_z)

    convolved_image = apply_otf(seed_array, otf)

    result = u * convolved_image
    # st()
    final_result_stack[:,:,N-ind-1] = result[:,:,N//2]
    # result = np.random.poisson(result)


    # tf.imshow(np.swapaxes(u,0,2))
    # tf.imwrite('test1.tiff',np.swapaxes(u,0,2))
    # tf.imshow(np.swapaxes(convolved_image,0,2))
    # tf.imwrite('test2.tiff',np.swapaxes(convolved_image,0,2))
    # tf.imshow(np.swapaxes(result,0,2))
    # tf.imwrite('test3.tiff',np.swapaxes(result,0,2))

tf.imwrite('1.tiff',np.swapaxes(final_result_stack,0,2))
print('done')
# plt.show()
# st()
# vizualize

# mlab.figure()
# mlab.contour3d(u, contours=4, transparent=True)
# mlab.contour3d(beads_array, contours=4, transparent=True)
# mlab.figure()
# mlab.contour3d(u, contours=4, transparent=True)
# mlab.contour3d(convolved_image, contours=4, transparent=True)
# mlab.figure()
# mlab.contour3d(result, contours=4, transparent=True)
# mlab.show()

# # st()

# plt.subplot(1,2,1)
# plt.imshow(result[...,N//2].T)
# plt.title("yz slice")
# plt.subplot(1,2,2)
# plt.imshow(result[N//2,...])
# plt.title("xy slice")
# plt.show()