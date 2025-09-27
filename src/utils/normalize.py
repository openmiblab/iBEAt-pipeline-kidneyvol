import numpy as np
from skimage import measure
import trimesh
import vreg
from scipy.stats import skew as scipy_skew



# -------------------------------
# Dice coefficient
# -------------------------------
def dice_coefficient(a, b):
    a, b = a.astype(bool), b.astype(bool)
    inter = np.logical_and(a, b).sum()
    return 2.0 * inter / (a.sum() + b.sum() + 1e-9)


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



def projection_skewness(mask, centroid, axis, voxel_size, radius=10):
    """
    Compute skewness of voxel projections along an axis from a 3D binary mask,
    using only voxels at a given perpendicular distance (± tol) from the axis.

    Parameters
    ----------
    mask : (X,Y,Z) array, bool or int
        3D binary image (nonzero = inside kidney).
    centroid : (3,) array
        A point on the axis (in voxel/world coordinates).
    axis : (3,) array
        Axis direction vector (e.g. eigenvector).
    voxel_size : tuple of 3 floats
        Physical size (mm) of a voxel along (x,y,z).
    radius : float
        Target perpendicular distance from axis in physical units (mm).

    Returns
    -------
    skew : float
        Skewness of projection values.
    """
    # find all voxel coordinates (in index space)
    idxs = np.argwhere(mask > 0)  # (N,3) in (x,y,z)

    # convert to world coordinates
    coords = idxs.astype(float) * voxel_size  # reorder to (x,y,z)

    c = np.asarray(centroid, dtype=float).reshape(3)
    e = np.asarray(axis, dtype=float).reshape(3)

    # normalize axis
    e = e / np.linalg.norm(e)

    # vectors from centroid to points
    v = coords - c[None, :]
    # projections along axis
    t = v.dot(e)
    # perpendicular distance
    perp = v - np.outer(t, e)
    dist = np.linalg.norm(perp, axis=1)

    # select radial band
    t_sel = t[dist < radius] # Projection values for selected voxels.
    # sel_idxs = idxs[mask_band] # indices of selected voxels - not needed here.
    n = t_sel.size

    if n < 3:
        return 0
    else:
        return scipy_skew(t_sel)
    

def kidney_eigenvecs(mask, centroid, eigvecs, eigvals, voxel_size, radius=10):
    # Best so far but not perfect

    # Ensure that the order and direction of the eigenvectors is defined by intrinsic shape.
    # i=0: kidney sagittal (through the hilum pointing out)
    # i=1: kidney transversal (right hand rule - pointing right if you look into the hole)
    # i=2: kidney longitudinal (bottom to top pole)

    indices = [0,1,2]

    # Find eigenvector along FH (normally first - principal axis)
    # Point it to the top of the kidney assuming the y-axis is head to feet
    foot_to_head = [0,-1,0]
    # proj_foot_to_head = [abs(np.dot(eigvecs[:,i], foot_to_head)) for i in indices]
    # foot_to_head_axis_index = proj_foot_to_head.index(max(proj_foot_to_head))
    foot_to_head_axis_index = 0
    foot_to_head_axis = eigvecs[:, foot_to_head_axis_index].copy()
    if np.dot(foot_to_head_axis, foot_to_head) < 0:
        foot_to_head_axis *= -1

    indices.remove(foot_to_head_axis_index)

    # For the remaining axis, project points within a certain distance 
    # on the axis and compute skewness of the distribution. The idea 
    # is that the axis with most skewness is pointing through the hole. 
    # This is the secondary axis (kidney axial). The third is the transversal

    # Find eigenvector with maximal skewness (out of the hilum)
    skew = [projection_skewness(mask, centroid, eigvecs[:,i], voxel_size, radius) for i in indices]
    abs_skew = [np.abs(s) for s in skew]
    maximal_skewness_index = abs_skew.index(max(abs_skew))
    sagittal_axis_index = indices[maximal_skewness_index]
    sagittal_axis = eigvecs[:, sagittal_axis_index].copy()
    if skew[maximal_skewness_index] < 0:
        sagittal_axis *= -1

    # Find kidney transversal eigenvector
    transversal_axis = np.cross(foot_to_head_axis, sagittal_axis)

    # Build new eigenvectors
    eigvecs_std = np.zeros((3,3)) #  
    eigvecs_std[:,0] = sagittal_axis 
    eigvecs_std[:,1] = transversal_axis
    eigvecs_std[:,2] = foot_to_head_axis

    return eigvecs_std
    

def volume_normalize_mask(mask, voxel_size=(1.0,1.0,1.0), target_spacing=1.0, target_volume=1e6, mirror_axis=None):
    """
    Normalize a 3D binary mask using mesh-based PCA alignment and scaling.
    Centers the mesh in the middle of the volume grid.
    """
    # TODO: either in this function or another compute scale invariant 
    # shape features such as eigenvalues of normalized volumes. Also 
    # rotation angles versus patient reference frame (obliqueness).

    # voxel size in mm
    # target_volume in mm3

    # Optional mirroring
    if mirror_axis is not None:
        mask = np.flip(mask, mirror_axis)

    # Build volume with identity affine and a corner on the origina
    volume = vreg.volume(mask.astype(float), spacing=voxel_size)

    # Align principal axes to reference frame
    centroid, eigvecs, eigvals = inertia_principal_axes(mask, voxel_size)
    # eigvecs = kidney_eigenvecs_v1(eigvecs)
    eigvecs = kidney_eigenvecs(mask, centroid, eigvecs, eigvals, voxel_size, radius=10)
    new_affine = pca_affine(volume.affine, centroid, eigvecs)
    volume.set_affine(new_affine)

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
        "original_shape": mask.shape,
        "voxel_size": voxel_size,
        "centroid": centroid,
        "scale": scale,
    }

    return mask_norm, params

# -------------------------
# Normalize mask
# -------------------------

def mask_to_mesh(mask, spacing=(1.0, 1.0, 1.0)):
    """Convert 3D binary mask to triangular mesh."""
    verts, faces, normals, values = measure.marching_cubes(
        mask.astype(float), level=0.5, spacing=spacing
    )
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return mesh



def mesh_to_mask(mesh, shape, pitch):
    """
    Voxelize a mesh into a binary mask of given shape and voxel pitch.
    
    Args:
        mesh (trimesh.Trimesh): Input mesh.
        shape (tuple[int]): Shape of the output mask (z,y,x).
        pitch (float or tuple[float]): Voxel size in mm. Can be scalar or (dx,dy,dz).
    
    Returns:
        mask (ndarray): Binary mask of shape `shape`.
    """
    # If scalar pitch, convert to 3-tuple
    if np.isscalar(pitch):
        pitch = (pitch, pitch, pitch)
    pitch = np.array(pitch, dtype=float)

    # Voxelize mesh
    voxelized = mesh.voxelized(pitch=min(pitch))  # voxelized() only accepts scalar pitch

    # Map voxel coordinates to the target grid
    coords = voxelized.points / pitch  # scale to original voxel dimensions
    coords = np.round(coords).astype(int)

    # Allocate mask
    mask = np.zeros(shape, dtype=bool)

    # Clip to valid indices
    valid = (
        (coords[:, 0] >= 0) & (coords[:, 0] < shape[0]) &
        (coords[:, 1] >= 0) & (coords[:, 1] < shape[1]) &
        (coords[:, 2] >= 0) & (coords[:, 2] < shape[2])
    )
    coords = coords[valid]
    mask[coords[:, 0], coords[:, 1], coords[:, 2]] = True

    return mask

def normalize_mask(mask, voxel_size=(1.0,1.0,1.0), target_volume=1e6, mirror_axis=None):
    """
    Normalize a 3D binary mask using mesh-based PCA alignment and scaling.
    Centers the mesh in the middle of the volume grid.
    """
    voxel_size = np.array(voxel_size, dtype=float)
    original_shape = np.array(mask.shape, dtype=float)

    # Step 1: Mask → Mesh
    mesh = mask_to_mesh(mask, spacing=voxel_size)

    # # Step 2: Center at origin
    centroid = mesh.vertices.mean(axis=0)
    mesh.vertices -= centroid

    # Step 2: PCA alignment
    cov = np.cov(mesh.vertices.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    rotation = eigvecs[:, idx]
    mesh.vertices = mesh.vertices @ rotation

    # Step 3: Scale based on physical volume
    voxel_volume = np.prod(voxel_size)
    current_volume = mask.sum() * voxel_volume
    scale = (target_volume / current_volume) ** (1/3)
    mesh.vertices *= scale

    # Step 4: Optional mirroring
    if mirror_axis is not None:
        mesh.vertices[:, mirror_axis] *= -1

    # Step 5: Translate so mesh is centered in the grid
    volume_center = voxel_size * original_shape / 2.0
    mesh.vertices += volume_center

    # Step 6: Build normalized mesh and rasterize back
    iso_pitch = float(min(voxel_size))
    shape_iso = np.ceil(original_shape * voxel_size / iso_pitch).astype(int)
    mask_norm = mesh_to_mask(mesh, shape_iso, iso_pitch)

    params = {
        "original_shape": mask.shape,
        "voxel_size": voxel_size,
        "centroid": centroid,
        "scale": scale,
        "rotation": rotation,
        "translation": volume_center,
        "mirrored_axis": mirror_axis,
        'iso_pitch': iso_pitch,
    }

    return mask_norm, params


def denormalize_mask(mask_norm, params):
    """
    Reverse the normalization of a 3D mask.

    Args:
        mask_norm (ndarray): Normalized binary mask.
        params (dict): Dictionary returned by normalize_mask.

    Returns:
        mask_recon (ndarray): Mask in the original space.
    """
    
    # Step 1: Mask → Mesh
    iso_pitch = params['iso_pitch']
    mesh = mask_to_mesh(mask_norm, spacing = 3 * [iso_pitch])

    # Step 2: Translate mesh back from grid center
    mesh.vertices -= params['translation']

    # Step 3: Undo mirroring
    if params['mirrored_axis'] is not None:
        mesh.vertices[:, params['mirrored_axis']] *= -1

    # Step 4: Undo scaling
    mesh.vertices /= params['scale']

    # Step 5: Undo PCA rotation
    mesh.vertices = mesh.vertices @ params['rotation'].T

    # Step 6: Add original centroid
    mesh.vertices += params['centroid']

    # Step 7: Rasterize back to original mask shape with original voxel_size
    mask_recon = mesh_to_mask(mesh, shape=params['original_shape'], pitch=params['voxel_size'])

    return mask_recon
