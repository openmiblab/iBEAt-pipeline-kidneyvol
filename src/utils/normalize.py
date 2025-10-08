import numpy as np
from skimage import measure
import trimesh
import vreg
from scipy.stats import skew as scipy_skew



# -------------------------------
# Dice coefficient
# -------------------------------


def covariance(x, y):
    """
    Compute the covariance between two 1D vectors x and y.

    Parameters
    ----------
    x : array-like, shape (n,)
    y : array-like, shape (n,)

    Returns
    -------
    cov : float
        Covariance between x and y.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    n = x.size
    cov = np.sum((x - x.mean()) * (y - y.mean())) / (n - 1)
    return cov


def dice_coefficient(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute the Dice similarity coefficient between two binary masks.

    Parameters
    ----------
    mask1 : np.ndarray
        First binary mask (values should be 0 or 1).
    mask2 : np.ndarray
        Second binary mask (values should be 0 or 1).

    Returns
    -------
    float
        Dice coefficient, ranging from 0 (no overlap) to 1 (perfect overlap).

    Notes
    -----
    The Dice coefficient is defined as:
        Dice = 2 * |A ∩ B| / (|A| + |B|)
    """
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    intersection = np.logical_and(mask1, mask2).sum()
    size_sum = mask1.sum() + mask2.sum()

    if size_sum == 0:
        return 1.0  # Both masks empty → perfect similarity
    return 2.0 * intersection / size_sum



def inertia_principal_axes(volume, voxel_size=(1.0,1.0,1.0), eps=1e-12):
    """
    Compute intensity-weighted center of mass and inertia (second-moment) tensor,
    and return principal axes (eigenvectors) and eigenvalues.

    Args:
        volume (ndarray): 3D numpy array (x,y,z) of intensities (non-negative typically).
        voxel_size (tuple[float] or np.ndarray): physical voxel spacing (dx, dy, dz).
        eps (float): small value to avoid division by zero.

    Returns:
        centroid_phys (ndarray shape (3,)): intensity-weighted centroid in physical coordinates (x,y,z).
        eigvals (ndarray shape (3,)): eigenvalues (descending).
        eigvecs (ndarray shape (3,3)): eigenvectors as columns; eigvecs[:,i] is eigenvector for eigvals[i].
        inertia (ndarray shape (3,3)): the computed inertia / second-moment matrix used.
    """
    voxel_size = np.asarray(voxel_size, dtype=float)
    if voxel_size.size != 3:
        raise ValueError("voxel_size must be length-3 (dx,dy,dz)")

    # Get indices and intensities for non-zero (or all) voxels
    # Using all voxels is fine; we can optionally ignore zeros if desired by the user upstream.
    coords_idx = np.argwhere(volume != 0)   # indices in (x,y,z) order
    if coords_idx.size == 0:
        raise ValueError("Volume contains no nonzero voxels.")

    intensities = volume[coords_idx[:,0], coords_idx[:,1], coords_idx[:,2]].astype(float)
    total_mass = intensities.sum()
    if total_mass <= eps:
        raise ValueError("Total intensity (mass) is zero or too small.")

    coords_phys = coords_idx.astype(float) * voxel_size  # shape (N,3) in (x,y,z)

    # Intensity-weighted centroid (center of mass)
    centroid_phys = (coords_phys * intensities[:,None]).sum(axis=0) / total_mass

    # Centralized coordinates
    delta = coords_phys - centroid_phys  # (N,3)

    # Compute the 3x3 inertia / second moment matrix (covariance-like, but with mass)
    # We compute the second central moments: M = sum_i w_i * (delta_i ⊗ delta_i)
    # This is essentially the (unnormalized) weighted covariance multiplied by total_mass.
    # Optionally normalize by total_mass to get weighted covariance; eigenvectors same either way.
    M = np.einsum('ni,nj->ij', delta * intensities[:,None], delta)  # shape (3,3)

    # If you prefer the covariance form: M_cov = M / total_mass
    # For axis extraction eigenvectors are identical up to eigenvalue scaling.
    # We'll compute eigen-decomposition of M (symmetric)
    # Use np.linalg.eigh (for symmetric matrices)
    eigvals_raw, eigvecs_raw = np.linalg.eigh(M)  # ascending order

    # Sort descending
    idx = np.argsort(eigvals_raw)[::-1]
    eigvals = eigvals_raw[idx]
    eigvecs = eigvecs_raw[:, idx]

    # Return inertia matrix as used (unnormalized). If user wants covariance: M / total_mass
    return centroid_phys, eigvecs, eigvals


def pca_affine(original_affine, centroid, eigvecs):
    """
    Build a new affine aligned to PCA axes.

    Args:
        original_affine (4x4): Input affine (voxel -> world).
        centroid (3,): PCA centroid in world coords.
        eigvecs (3x3): PCA eigenvectors, columns = axes in world.

    Returns:
        new_affine (4x4): Voxel -> PCA-aligned coords.
        transform_world_to_pca (4x4): Extra transform applied in world space.
    """
    # World -> PCA coords
    R = eigvecs.T   # rotation
    T = -centroid   # translation

    # Build 4x4 homogeneous transform world->PCA
    W2P = np.eye(4)
    W2P[:3,:3] = R
    W2P[:3,3] = R @ T

    return W2P @ original_affine



def second_vector_cone_dir(mask, centroid, eigvec1, voxel_size, cone_angle=10):
    """Orient the eigenvector int he direction of
    the half-cone (defined by angular aperture around that axis) with the least mass.

    This function works but is not used at the moment as the cone sweep appears 
    prefereable. Could be reinstated if a decision is made later on to stick to 
    eigenvectors after all.

    Args:
        mask (numpy.ndarray): Binary mask (3D volume).
        centroid (numpy.ndarray): 3-element centroid.
        eigvec1 (numpy.ndarray): In-plane unit vector 1.
        voxel_size (numpy.ndarray): 3-element voxel dimensions.
        cone_angle (float, optional): Cone aperture (degrees). Defaults to 10.

    Returns:
        numpy.ndarray: Unit vector (3,) in the plane, pointing toward the lighter half.
    """
    # normalize eigenvectors
    eigvec1 = eigvec1 / np.linalg.norm(eigvec1)

    # voxel coordinates in physical units, shifted to centroid
    coords = np.argwhere(mask > 0).astype(float) * voxel_size
    coords -= centroid

    if coords.shape[0] == 0:
        return eigvec1  # empty mask

    cos_cone = np.cos(np.deg2rad(cone_angle / 2))  # use half-angle for selection

    # Sweep candidate directions in plane
    d = eigvec1

    # normalize coords for angle test
    norms = np.linalg.norm(coords, axis=1)
    valid = norms > 0
    unit_coords = np.zeros_like(coords)
    unit_coords[valid] = coords[valid] / norms[valid, None]

    # angle test: voxel inside cone if dot(u, d) >= cos(theta)
    dots = np.abs(unit_coords @ d)
    cone_voxels = coords[dots >= cos_cone]

    # project onto candidate axis
    proj = cone_voxels @ d

    left_mass = np.sum(proj < 0)
    right_mass = np.sum(proj >= 0)

    best_vec = -d if left_mass < right_mass else d

    return best_vec



def second_vector_cone_sweep(mask, centroid, eigvec1, eigvec2, voxel_size, cone_angle=10, angle_step=1):
    """Find a direction lying in the plane of eigvec1 & eigvec2 that points to
    the half-cone (defined by angular aperture around that axis) with the least mass.

    If multiple directions have the same minimum mass, the middle candidate is chosen.

    Args:
        mask (numpy.ndarray): Binary mask (3D volume).
        centroid (numpy.ndarray): 3-element centroid.
        eigvec1 (numpy.ndarray): In-plane unit vector 1.
        eigvec2 (numpy.ndarray): In-plane unit vector 2.
        voxel_size (numpy.ndarray): 3-element voxel dimensions.
        cone_angle (float, optional): Cone aperture (degrees). Defaults to 30.
        angle_step (float, optional): Step size (degrees) for candidate sweep. Defaults to 1.

    Returns:
        numpy.ndarray: Unit vector (3,) in the plane, pointing toward the lighter half.
    """
    # normalize eigenvectors
    eigvec1 = eigvec1 / np.linalg.norm(eigvec1)
    eigvec2 = eigvec2 / np.linalg.norm(eigvec2)

    # voxel coordinates in physical units, shifted to centroid
    coords = np.argwhere(mask > 0).astype(float) * voxel_size
    coords -= centroid

    if coords.shape[0] == 0:
        return eigvec1  # empty mask

    cos_cone = np.cos(np.deg2rad(cone_angle / 2))  # use half-angle for selection

    candidates = []  # store (lighter_mass, angle, direction)

    # Sweep candidate directions in plane
    for angle in np.arange(0, 180, angle_step):
        rad = np.deg2rad(angle)
        d = np.cos(rad) * eigvec1 + np.sin(rad) * eigvec2
        d /= np.linalg.norm(d)

        # normalize coords for angle test
        norms = np.linalg.norm(coords, axis=1)
        valid = norms > 0
        unit_coords = np.zeros_like(coords)
        unit_coords[valid] = coords[valid] / norms[valid, None]

        # angle test: voxel inside cone if dot(u, d) >= cos(theta)
        dots = np.abs(unit_coords @ d)
        cone_voxels = coords[dots >= cos_cone]

        if cone_voxels.shape[0] == 0:
            continue  # skip empty cone

        # project onto candidate axis
        proj = cone_voxels @ d

        left_mass = np.sum(proj < 0)
        right_mass = np.sum(proj >= 0)
        lighter_mass = min(left_mass, right_mass)

        # orient toward lighter side
        d_oriented = -d if left_mass < right_mass else d
        candidates.append((lighter_mass, angle, d_oriented))

    if not candidates:
        return eigvec1

    # find minimum lighter_mass
    min_mass = min(c[0] for c in candidates)

    # filter candidates with min_mass
    min_candidates = [c for c in candidates if c[0] == min_mass]

    # choose the middle one by angle
    min_candidates.sort(key=lambda x: x[1])
    middle_idx = len(min_candidates) // 2
    best_vec = min_candidates[middle_idx][2]

    return best_vec



def kidney_canonical_axes(mask, centroid, eigvecs, voxel_size):
    """
    Returns orthogonal vectors in a canonical reference frame optimized 
    for normalization of kidney shapes.

    - The z-axis (kidney longitudinal) is the principal eigenvector oriented fro  head to foot
    - The x-axis (kidney sagittal) is perpendicular to the longitudinal axis, through the 
      centroid and pointing in the direction with least mass in a small cone around the axis
    - The y-axis (kidney transverse) is derived from the first two so as to make a right-handed 
      X,Y,Z Cartesian reference frame
    """

    # Find eigenvector along FH 
    foot_to_head = [0,-1,0]
    longitudinal_axis = eigvecs[:, 0].copy()
    if np.dot(longitudinal_axis, foot_to_head) < 0:
        longitudinal_axis *= -1

    # Decide direction of the secondary axes
    sagittal_axis = second_vector_cone_sweep(mask, centroid, eigvecs[:,1], eigvecs[:,2], voxel_size)
    transversal_axis = np.cross(longitudinal_axis, sagittal_axis)

    # Build new eigenvectors
    canonical_axes = np.zeros((3,3))  
    canonical_axes[:,0] = sagittal_axis 
    canonical_axes[:,1] = transversal_axis
    canonical_axes[:,2] = longitudinal_axis

    return canonical_axes


def normalize_kidney_mask(mask, voxel_size, side):
    """
    Normalize a 3D binary mask using mesh-based PCA alignment and scaling.
    Centers the mesh in the middle of the volume grid.
    """
    # TODO: either in this function or another compute scale invariant 
    # shape features such as eigenvalues of normalized volumes. Also 
    # rotation angles versus patient reference frame (obliqueness).

    if side not in ['left', 'right']:
        raise ValueError(
            f"The side argument must be either 'left' or 'right'. "
            f"You have entered {side}. "
        )
    if np.size(voxel_size) != 3:
        raise ValueError("Voxel size must have 3 elements.")
    if np.ndim(mask) != 3:
        raise ValueError("mask must be 3D. ")
 
    # voxel size in mm
    # target_volume in mm3
    target_spacing = 1.0
    target_volume = 1e6

    # Optional mirroring
    if side == 'left':
        mask = np.flip(mask, 0)

    # Build volume with identity affine and a corner on the origina
    volume = vreg.volume(mask.astype(float), spacing=voxel_size)

    # Align principal axes to reference frame
    centroid, eigvecs, eigvals = inertia_principal_axes(mask, voxel_size)
    canonical_axes = kidney_canonical_axes(mask, centroid, eigvecs, voxel_size)
    canonical_affine = pca_affine(volume.affine, centroid, canonical_axes)
    volume.set_affine(canonical_affine)

    # Scale to target volume
    voxel_volume = np.prod(voxel_size)
    current_volume = mask.sum() * voxel_volume
    scale = (target_volume / current_volume) ** (1/3)
    volume = volume.stretch(scale)

    # Resample on standard isotropic volume
    target_length = 3 * (target_volume ** (1/3)) # length in mm
    target_dim = np.ceil(1 + target_length / target_spacing).astype(int) # length in mm
    target_shape = 3 * [target_dim]
    target_affine = np.diag(3 * [target_spacing] + [1.0])
    target_affine[:3,3] = 3 * [- target_spacing * (target_dim - 1) / 2]
    volume = volume.slice_like((target_shape, target_affine))

    # Convert back to mask 
    mask_norm = volume.values > 0.5

    params = {
        "centroid": centroid,
        "eigvecs": eigvecs,
        "eigvals": eigvals,
        "scale": scale,
        "canonical_axes": canonical_axes,
        "canonical_affine": canonical_affine,
        "scale": scale,
        "output_affine": target_affine,
    }

    return mask_norm, params

