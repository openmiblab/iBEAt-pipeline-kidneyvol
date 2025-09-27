import numpy as np


from skimage import measure, morphology
from scipy.spatial import cKDTree
from scipy.ndimage import binary_fill_holes

from skimage import measure
import trimesh
import pyshtools as pysh


# ----------------------
# Utilities / preprocessing
# ----------------------

def preprocess_volume(vol, min_size=1000, closing_radius=2):
    """Simple morphological cleanup on binary volume."""
    vol = vol.astype(bool)
    vol = morphology.remove_small_objects(vol, min_size=min_size)
    if closing_radius > 0:
        vol = morphology.binary_closing(vol, morphology.ball(closing_radius))
    vol = binary_fill_holes(vol)
    return vol

def extract_mesh_from_volume(vol, spacing=(1.0,1.0,1.0), level=0.5):
    """
    Extract triangle mesh from binary volume using marching cubes.
    Returns verts (N,3) and faces (M,3).
    spacing: voxel spacing tuple (z,y,x) or (dx,dy,dz) according to skimage usage.
    """
    verts, faces, normals, values = measure.marching_cubes(vol.astype(np.uint8), level=level, spacing=spacing)
    return verts, faces



def dice_coefficient(vol_a, vol_b):
    """
    Compute Dice coefficient between two binary volumes.
    """
    vol_a = vol_a.astype(bool)
    vol_b = vol_b.astype(bool)
    intersection = np.logical_and(vol_a, vol_b).sum()
    size_a = vol_a.sum()
    size_b = vol_b.sum()
    if size_a + size_b == 0:
        return 1.0
    return 2.0 * intersection / (size_a + size_b)

def surface_distances(vol_a, vol_b, spacing=(1.0,1.0,1.0)):
    """
    Compute surface distances (Hausdorff and mean) between two binary volumes.
    Args:
      vol_a, vol_b: binary 3D arrays
      spacing: voxel spacing (dz,dy,dx)
    Returns:
      hausdorff, mean_dist
    """
    # extract meshes
    verts_a, faces_a, _, _ = measure.marching_cubes(vol_a.astype(np.uint8), level=0.5, spacing=spacing)
    verts_b, faces_b, _, _ = measure.marching_cubes(vol_b.astype(np.uint8), level=0.5, spacing=spacing)

    # build kd-trees
    tree_a = cKDTree(verts_a)
    tree_b = cKDTree(verts_b)

    # distances from A→B and B→A
    d_ab, _ = tree_b.query(verts_a, k=1)
    d_ba, _ = tree_a.query(verts_b, k=1)

    hausdorff = max(d_ab.max(), d_ba.max())
    mean_dist = 0.5 * (d_ab.mean() + d_ba.mean())
    return hausdorff, mean_dist





def decompose(volume, lmax=15):
    # -----------------------
    # 1. Original volume
    # -----------------------

    # Compute centroid and max extent
    coords = np.argwhere(volume)
    centroid = coords.mean(axis=0)
    max_extent = (coords.max(axis=0) - coords.min(axis=0)).max() / 2.0

    # -----------------------
    # 2. Normalize coordinates to unit sphere
    # -----------------------
    norm_coords = (coords - centroid) / max_extent
    Xc, Yc, Zc = norm_coords[:,0], norm_coords[:,1], norm_coords[:,2]
    R = np.sqrt(Xc**2 + Yc**2 + Zc**2)
    Theta = np.arccos(np.clip(Zc / R, -1, 1))       # polar angle
    Phi = np.arctan2(Yc, Xc) % (2*np.pi)            # azimuth

    # -----------------------
    # 3. Map points to 2D spherical grid
    # -----------------------
    n_theta, n_phi = 64, 128
    # n_theta, n_phi = 128, 256 # n_phi must be n_theta or 2*n_theta
    radii_grid = np.zeros((n_theta, n_phi))

    # Fill grid with max radius in each cell
    for i in range(norm_coords.shape[0]):
        t_idx = int(Theta[i] / np.pi * (n_theta-1))
        p_idx = int(Phi[i] / (2*np.pi) * (n_phi-1))
        radii_grid[t_idx, p_idx] = max(radii_grid[t_idx, p_idx], R[i])

    # -----------------------
    # 4. Create SHGrid and compute coefficients
    # -----------------------
    grid = pysh.SHGrid.from_array(radii_grid)
    coeffs = grid.expand()  # SHRealCoeffs
    
    coeffs_array = coeffs.to_array()[:, :lmax+1, :lmax+1]
    coeffs_trunc = pysh.SHCoeffs.from_array(coeffs_array)

    return coeffs_trunc, centroid, max_extent


def reconstruct_surface_from_coeffs(coeffs, lmax = 8):

    # Suppose coeffs is your SHCoeffs object (from pysh)
    coeffs_array = coeffs.to_array()[:, :lmax+1, :lmax+1]
    coeffs_trunc = pysh.SHCoeffs.from_array(coeffs_array)

    # Reconstruct the spherical function
    grid_recon = coeffs_trunc.expand(grid='DH')  # gives SHGrid
    radii_recon = grid_recon.to_array()

    # -----------------------
    # 2. Convert back to 3D surface
    # -----------------------
    # Define the same theta/phi grid
    n_theta, n_phi = radii_recon.shape
    theta = np.linspace(0, np.pi, n_theta)       # polar
    phi = np.linspace(0, 2*np.pi, n_phi)         # azimuth
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

    # Convert spherical coords (r, theta, phi) -> Cartesian
    x = radii_recon * np.sin(theta_grid) * np.cos(phi_grid)
    y = radii_recon * np.sin(theta_grid) * np.sin(phi_grid)
    z = radii_recon * np.cos(theta_grid)

    # Flatten for point cloud
    points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

    # -----------------------
    # 3. Create mesh from point cloud
    # -----------------------
    # Use alpha shape or ball pivoting; simplest: convex hull for now
    recon_mesh = trimesh.convex.convex_hull(points)

    # Save or show mesh
    recon_mesh.export("kidney_recon_lmax8.ply")



def reconstruct_volume_from_coeffs(coeffs, vol_shape, centroid, max_extent, lmax = 15):

    coeffs_array = coeffs.to_array()[:, :lmax+1, :lmax+1]
    coeffs_trunc = pysh.SHCoeffs.from_array(coeffs_array)

    # Assume coeffs is SHCoeffs object (from earlier steps)
    grid_recon = coeffs_trunc.expand(grid='DH')
    radii_recon = grid_recon.to_array()  # normalized unit-sphere radii

    # -----------------------
    # 4. Reconstruct 3D volume with original shape
    # -----------------------
    nx, ny, nz = vol_shape
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Coordinates relative to original centroid
    Xc = (X - centroid[0]) / max_extent
    Yc = (Y - centroid[1]) / max_extent
    Zc = (Z - centroid[2]) / max_extent

    R = np.sqrt(Xc**2 + Yc**2 + Zc**2)
    Theta = np.arccos(np.clip(Zc / (R + 1e-9), -1, 1))
    Phi = np.arctan2(Yc, Xc) % (2*np.pi)

    # Map spherical coordinates to SHGrid indices
    n_theta, n_phi = radii_recon.shape
    theta_idx = (Theta / np.pi * (n_theta-1)).astype(int)
    phi_idx   = (Phi / (2*np.pi) * (n_phi-1)).astype(int)
    r_boundary = radii_recon[theta_idx, phi_idx]

    # Binary reconstruction in original grid
    volume_recon = (R <= r_boundary).astype(np.uint8)

    return volume_recon


def power_spectrum(coeffs):
    # Real and rotationally invariant

    # Suppose we already computed coeffs from SH expansion (see previous example)
    # coeffs is an SHCoeffs object from pyshtools

    # Convert coefficients to array: shape (2, lmax+1, lmax+1)
    # axis 0: [0]=real part, [1]=imag part
    coeffs_array = coeffs.to_array()

    # -----------------------
    # 1. Compute rotation-invariant power spectrum
    # -----------------------
    lmax = coeffs.lmax
    power_spectrum = []

    for l in range(lmax + 1):
        # Get coefficients for this degree l across all orders m
        c_l = coeffs_array[:, l, :l+1]  # shape (2, l+1)
        # Combine real + imaginary into complex
        c_l_complex = c_l[0] + 1j * c_l[1]
        # Sum of squared magnitudes across m
        P_l = np.sqrt(np.sum(np.abs(c_l_complex)**2))
        power_spectrum.append(P_l)

    power_spectrum = np.array(power_spectrum)

    return power_spectrum


def cosine_similarity(a, b):
    # Simularity between two descriptor vectors - distance between shapes
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def reconstruct_shape(vol_orig, spacing, lmax=200):

    # 1. Fit spherical harmonics
    coeffs, centroid, max_extent = decompose(vol_orig, lmax=lmax)

    # coeffs contains complex SH coefficients = Fourier shape descriptors
    # -----------------------
    # 6. Use real descriptors for comparison
    # -----------------------
    descriptor_vector_real = coeffs.to_array().ravel().real
    descriptor_vector_power = power_spectrum(coeffs)

    vol_rec = reconstruct_volume_from_coeffs(coeffs, vol_orig.shape, centroid, max_extent, lmax=lmax)

    # 4. Compute evaluation metrics
    dice = dice_coefficient(vol_orig, vol_rec)
    hausdorff, mean_dist = surface_distances(vol_orig, vol_rec, spacing=spacing)

    print('Dice: ', dice)

    return vol_rec, {'p1': [1, 'par1', '%', 'float'], 'p2': [1, 'par2', '%', 'float']}