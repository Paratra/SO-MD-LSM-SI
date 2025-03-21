import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace as st
from scipy.ndimage import convolve
from numpy.fft import fftn, fftshift, ifftn



class Ellipsoid:
    '''
    example: 
        ellipsoid = Ellipsoid(num_spheres=500, elli_axis=(200, 105, 105), spheres_radius=5, grid_size=512, seed=42)
        grid = ellipsoid.get_sample()
    '''

    def __init__(self, num_spheres, elli_axis, spheres_radius, grid_size, sample_reIndex=1.5, medium_reIndex=1.33, seed=42):

        self.num_spheres = num_spheres # Number of spheres to generate
        (self.a, self.b, self.c) = elli_axis  # Semi-axes of the ellipsoid within the 512 grid
        self.radius = spheres_radius  # Radius of spheres in pixels
        self.grid_size = grid_size  # Size of the ndarray
        self.seed = seed # random seed for reproducibility
        # Initialize the grid
        self.grid = np.zeros((grid_size, grid_size, grid_size))
        self.signal = np.copy(self.grid)
        self.sample_reIndex = sample_reIndex
        self.medium_reIndex = medium_reIndex

    def in_ellipsoid(self, x, y, z):
        """Check if a point (x, y, z) is inside the ellipsoid defined by semi-axes (a, b, c)."""
        return (x/self.a)**2 + (y/self.b)**2 + (z/self.c)**2 <= 1

    def draw_sphere(self, center):
        """Draw a sphere in a 3D grid at specified center with given radius."""
        x0, y0, z0 = center
        for x in range(max(0, x0-self.radius), min(self.grid_size, x0+self.radius+1)):
            for y in range(max(0, y0-self.radius), min(self.grid_size, y0+self.radius+1)):
                for z in range(max(0, z0-self.radius), min(self.grid_size, z0+self.radius+1)):
                    if (x-x0)**2 + (y-y0)**2 + (z-z0)**2 <= self.radius**2:
                        self.grid[x, y, z] = self.sample_reIndex
                        self.signal[x, y, z] = self.medium_reIndex

    def get_sample(self):
        # Generate sphere centers randomly inside the ellipsoid
        if self.seed == 'None':
            pass
        else:
            np.random.seed(self.seed)  # for reproducibility
            
        for ind in range(self.num_spheres):
            while True:
                # print(ind)
                # Generate random center within the grid
                x = np.random.randint(0, self.grid_size)
                y = np.random.randint(0, self.grid_size)
                z = np.random.randint(0, self.grid_size)
                # Adjust centers to be relative to the center of the grid
                if self.in_ellipsoid(x - self.grid_size//2, y - self.grid_size//2, z - self.grid_size//2):
                    self.draw_sphere((x, y, z))
                    break

        return self.grid, self.signal




def rotate_wavefront(u0, angle, axis='x'):
    """
    Rotates the wavefront around a specified axis by a given angle.

    Parameters:
        u0 (np.ndarray): The initial wavefront array.
        angle (float): Rotation angle in degrees.
        axis (str): Axis to rotate around ('x', 'y', or 'z').

    Returns:
        np.ndarray: Rotated wavefront.
    """
    from scipy.ndimage import rotate
    if axis == 'x':
        return rotate(u0, angle, axes=(1,2), reshape=False)
    elif axis == 'y':
        return rotate(u0, angle, axes=(0,2), reshape=False)
    elif axis == 'z':
        return rotate(u0, angle, axes=(0,1), reshape=False)
    else:
        raise ValueError("Invalid axis specified. Choose 'x', 'y', or 'z'.")






def gaussian_otf_3d_anisotropic(na, wavelength, refractive_index, grid_size, pixel_size_xy, pixel_size_z):
    """
    Generate an anisotropic 3D Gaussian PSF based on NA, wavelength, and refractive index.
    
    Parameters:
    na (float): Numerical Aperture of the optical system.
    wavelength (float): Wavelength of light in the same units as pixel sizes.
    refractive_index (float): Refractive index of the medium.
    grid_size (tuple): The dimensions of the grid to calculate the PSF over (should be odd).
    pixel_size_xy (float): The size of each pixel in the xy-plane in the same units as wavelength.
    pixel_size_z (float): The size of each pixel along the z-axis in the same units as wavelength.

    Returns:
    np.ndarray: 3D Gaussian PSF array.
    """
    # Calculate the FWHM based on the lateral and axial resolution formulas
    fwhm_xy = 0.61 * wavelength / na
    fwhm_z = 2 * wavelength * refractive_index / (na ** 2)
    
    # Convert FWHM to standard deviations
    sigma_xy = fwhm_xy / 2.355
    sigma_z = fwhm_z / 2.355
    
    # Convert sigma to pixels
    sigma_xy_pixels = sigma_xy / pixel_size_xy
    sigma_z_pixels = sigma_z / pixel_size_z
    
    # Create a grid of x, y, z values centered at 0
    x = np.linspace(-int(grid_size[0]/2), int(grid_size[0]/2), grid_size[0])
    y = np.linspace(-int(grid_size[1]/2), int(grid_size[1]/2), grid_size[1])
    z = np.linspace(-int(grid_size[2]/2), int(grid_size[2]/2), grid_size[2])
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    
    # Gaussian formula with anisotropy
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma_xy_pixels**2) - (zz**2) / (2 * sigma_z_pixels**2))
    
    # Normalize the PSF so that the sum is 1
    psf /= np.sum(psf)
    otf = fftn(fftshift(psf))
    return otf


def apply_otf(image, otf):
    # Compute the Fourier transform of the image
    img_freq = fftn(image)
    
    # Multiply the image in the frequency domain by the OTF
    convolved_freq = img_freq * otf
    
    # Bring the convolved image back to the spatial domain
    convolved_image = ifftn(convolved_freq)
    return np.abs(convolved_image)  # Take the absolute value if the result is complex



def generate_beads_array(num_beads, space_dim, bead_value, iph=1000):
    """
    Generate a 3D numpy array with specified number of elements set to a certain value,
    and return the coordinates of these elements.

    Parameters:
    num_beads (int): Number of elements to set to the bead_value.
    space_dim (tuple): Dimensions of the 3D space (x, y, z).
    bead_value (float): Value to assign to the specified number of elements.

    Returns:
    np.ndarray: 3D numpy array with specific elements set to bead_value.
    list of tuples: List of coordinates where the array elements have been set to bead_value.
    """
    # Create an empty array with all elements set to zero
    array = np.zeros(space_dim)
    array_base = np.zeros(space_dim)
    
    # Generate random indices
    indices = np.random.choice(np.prod(space_dim), size=num_beads, replace=False)
    
    # Convert linear indices to 3D indices and set the value
    np.put(array, indices, bead_value)
    np.put(array_base, indices, iph)
    
    # Calculate 3D coordinates from linear indices
    coords = np.unravel_index(indices, space_dim)
    coords = list(zip(*coords))  # Create a list of tuples for coordinates
    
    return array, array_base, coords