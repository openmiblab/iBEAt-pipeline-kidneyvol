import numpy as np
import pyvista as pv
import time
from skimage import measure


def rotation_vector_to_matrix(rot_vec):
    """Convert a rotation vector (axis-angle) to a 3x3 rotation matrix using Rodrigues' formula."""
    theta = np.linalg.norm(rot_vec)
    if theta < 1e-8:
        return np.eye(3)
    k = rot_vec / theta
    K = np.array([[    0, -k[2],  k[1]],
                  [ k[2],     0, -k[0]],
                  [-k[1],  k[0],     0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


def ellipsoid_mask(shape, voxel_sizes=(1.0,1.0,1.0), center=(0,0,0), radii=(1,1,1), rot_vec=None):
    """
    Generate a 3D mask array with a rotated ellipsoid.

    Parameters
    ----------
    shape : tuple of int
        Shape of the 3D array (z, y, x).
    voxel_sizes : tuple of float
        Physical voxel sizes (dz, dy, dx).
    center : tuple of float
        Center of the ellipsoid in physical units, with (0,0,0) at the **middle of the volume**.
    radii : tuple of float
        Radii of the ellipsoid in physical units.
    rot_vec : array-like of shape (3,), optional
        Rotation vector. Magnitude = angle in radians, direction = rotation axis.

    Returns
    -------
    mask : np.ndarray of bool
        Boolean 3D mask with the ellipsoid.
    """
    dz, dy, dx = voxel_sizes
    zdim, ydim, xdim = shape

    # Rotation matrix
    if rot_vec is None:
        rotation = np.eye(3)
    else:
        rot_vec = np.array(rot_vec, dtype=float)
        rotation = rotation_vector_to_matrix(rot_vec)

    rz, ry, rx = radii
    D = np.diag([1/rz**2, 1/ry**2, 1/rx**2])
    A = rotation @ D @ rotation.T

    # Generate coordinate grids centered at 0
    z = (np.arange(zdim) - zdim/2 + 0.5) * dz
    y = (np.arange(ydim) - ydim/2 + 0.5) * dy
    x = (np.arange(xdim) - xdim/2 + 0.5) * dx
    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')

    coords = np.stack([zz - center[0], yy - center[1], xx - center[2]], axis=-1)
    vals = np.einsum('...i,ij,...j->...', coords, A, coords)

    return vals <= 1.0



def add_axes(p, xlabel="X", ylabel="Y", zlabel="Z", color=("red", "green", "blue")):

    # Draw your volume/mesh
    # p.add_volume(grid)

    # Custom axis length
    L = 50

    # Unit right-handed basis
    origin = np.array([0,0,0])
    X = np.array([1,0,0])
    Y = np.array([0,1,0])
    Z = np.cross(X, Y)   # guaranteed right-handed

    # Arrows
    p.add_mesh(pv.Arrow(start=origin, direction=X, scale=L), color=color[0])
    p.add_mesh(pv.Arrow(start=origin, direction=Y, scale=L), color=color[1])
    p.add_mesh(pv.Arrow(start=origin, direction=Z, scale=L), color=color[2])

    # Add 3D text at the arrow tips
    p.add_point_labels([origin + X*L], [xlabel], font_size=20, text_color=color[0], point_size=0)
    p.add_point_labels([origin + Y*L], [ylabel], font_size=20, text_color=color[1], point_size=0)
    p.add_point_labels([origin + Z*L], [zlabel], font_size=20, text_color=color[2], point_size=0)




def visualize_surface_reconstruction(original_mesh, reconstructed_mesh, opacity=(0.3,0.3)):
    # Convert trimesh to pyvista PolyData
    def trimesh_to_pv(mesh):
        faces = np.hstack([np.full((len(mesh.faces),1), 3), mesh.faces]).astype(np.int64)
        return pv.PolyData(mesh.vertices, faces)
    
    original_pv = trimesh_to_pv(original_mesh)
    reconstructed_pv = trimesh_to_pv(reconstructed_mesh)

    plotter = pv.Plotter(window_size=(800,600))
    plotter.background_color = 'white'
    plotter.add_mesh(original_pv, color='red', opacity=opacity[0], label='Original')
    plotter.add_mesh(reconstructed_pv, color='blue', opacity=opacity[1], label='Reconstructed')
    plotter.add_legend()
    plotter.add_text("Original (Red) vs Reconstructed (Blue)", font_size=14)
    plotter.camera_position = 'iso'
    plotter.show()


def display_both_surfaces(mask, mask_recon):
    # ---------------------------
    # 7. Visualize with PyVista
    # ---------------------------
    # Original mesh
    grid_orig = pv.wrap(mask.astype(np.uint8))
    contour_orig = grid_orig.contour(isosurfaces=[0.5])

    # Reconstructed mesh
    grid_recon = pv.wrap(mask_recon.astype(np.uint8))
    contour_recon = grid_recon.contour(isosurfaces=[0.5])

    plotter = pv.Plotter(shape=(1,2))
    plotter.subplot(0,0)
    plotter.add_text("Original", font_size=12)
    plotter.add_mesh(contour_orig, color="lightblue")

    plotter.subplot(0,1)
    plotter.add_text("Reconstructed", font_size=12)
    plotter.add_mesh(contour_recon, color="salmon")

    plotter.show()

def display_volume(original_volume, voxel_size=(1.0,1.0,1.0)):
    """
    Uses pyvista to display a volume,
    with the reconstructed volume shown as a solid surface,
    and a wireframe showing the volume boundaries.
    """
    voxel_size = np.array(voxel_size, dtype=float)
    plotter = pv.Plotter(window_size=(800, 600))
    plotter.background_color = 'white'
    
    # Create the original volume mesh from the numpy array
    mesh = pv.wrap(original_volume)
    mesh.spacing = voxel_size

    # Contour surface
    surface = mesh.contour(isosurfaces=[0.5])
    
    # Add the original surface to the plotter with transparency
    plotter.add_mesh(
        surface, 
        color='blue', 
        opacity=0.5, 
        style='surface', 
        label='Original Volume'
    )

    # Add a wireframe box to show the volume boundaries
    bounds = mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    box = pv.Box(bounds=bounds)
    plotter.add_mesh(box, color='black', style='wireframe', line_width=2)

    plotter.add_legend()
    plotter.add_text('3D Volume', font_size=20)
    add_axes(plotter)
    
    # Set the camera position and show the plot
    plotter.camera_position = 'iso'
    plotter.show()


# def display_kidney_normalization_v1(kidney, kidney_norm, kidney_voxel_size=(1.0,1.0,1.0), kidney_norm_voxel_size=None, title='Kidney normalization'):
#     """
#     Visualize original and reconstructed 3D volumes in PyVista with correct proportions.
#     """
#     kidney_voxel_size = np.array(kidney_voxel_size, dtype=float)
#     if kidney_norm_voxel_size is None:
#         kidney_norm_voxel_size = kidney_voxel_size
#     plotter = pv.Plotter(window_size=(800,600), shape=(1,2))
#     plotter.background_color = 'white'

#     # Wrap original volume
#     orig_vol = pv.wrap(kidney.astype(float))
#     orig_vol.spacing = kidney_voxel_size
#     orig_surface = orig_vol.contour(isosurfaces=[0.5])
#     plotter.subplot(0,0)
#     plotter.add_text(f"{title} (original)", font_size=12)
#     plotter.add_mesh(
#         orig_surface, color='lightblue', opacity=1.0, style='surface',     
#         ambient=0.1,           # ambient term
#         diffuse=0.9,           # diffuse shading
#     )
#     # Add wireframe box around original volume
#     bounds = orig_vol.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
#     box = pv.Box(bounds=bounds)
#     plotter.add_mesh(box, color='black', style='wireframe', line_width=2)
#     add_axes(plotter, xlabel='L', ylabel='F', zlabel='P', color=("red", "green", "blue"))

#     # Wrap reconstructed volume
#     recon_vol = pv.wrap(kidney_norm.astype(float))
#     recon_vol.spacing = kidney_norm_voxel_size
#     recon_surface = recon_vol.contour(isosurfaces=[0.5])
#     plotter.subplot(0,1)
#     plotter.add_text(f"{title} (normalized)", font_size=12)
#     plotter.add_mesh(recon_surface, color='lightblue', opacity=1.0, style='surface',
#         ambient=0.1,           # ambient term
#         diffuse=0.9,           # diffuse shading
#     )
#     # Add wireframe box around original volume
#     bounds = recon_vol.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
#     box = pv.Box(bounds=bounds)
#     plotter.add_mesh(box, color='black', style='wireframe', line_width=2)
#     add_axes(plotter, xlabel='LO', ylabel='PO', zlabel='HO', color=("red", "blue", "green"))

#     # Set the camera position and show the plot
#     plotter.camera_position = 'iso'
#     plotter.show()


def display_kidney_normalization(kidney, kidney_norm, kidney_voxel_size=(1.0,1.0,1.0), kidney_norm_voxel_size=None, title='Kidney normalization'):
    """
    Visualize original and reconstructed 3D volumes in PyVista with correct proportions.
    """
    kidney_voxel_size = np.array(kidney_voxel_size, dtype=float)
    if kidney_norm_voxel_size is None:
        kidney_norm_voxel_size = kidney_voxel_size
    plotter = pv.Plotter(window_size=(800,600), shape=(1,2))
    plotter.background_color = 'white'

    # Wrap original volume
    orig_vol = pv.wrap(kidney.astype(float))
    orig_vol.spacing = kidney_voxel_size
    orig_surface = orig_vol.contour(isosurfaces=[0.5])
    plotter.subplot(0,0)
    plotter.add_text(f"{title} (original)", font_size=12)
    plotter.add_mesh(
        orig_surface, color='lightblue', opacity=1.0, style='surface',     
        ambient=0.1,           # ambient term
        diffuse=0.9,           # diffuse shading
    )
    # Add wireframe box around original volume
    bounds = orig_vol.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    box = pv.Box(bounds=bounds)
    plotter.add_mesh(box, color='black', style='wireframe', line_width=2)
    add_axes(plotter, xlabel='L', ylabel='F', zlabel='P', color=("red", "green", "blue"))

    # Wrap reconstructed volume
    recon_vol = pv.wrap(kidney_norm.astype(float))
    recon_vol.spacing = kidney_norm_voxel_size
    recon_surface = recon_vol.contour(isosurfaces=[0.5])
    plotter.subplot(0,1)
    plotter.add_text(f"{title} (normalized)", font_size=12)
    plotter.add_mesh(recon_surface, color='lightblue', opacity=1.0, style='surface',
        ambient=0.1,           # ambient term
        diffuse=0.9,           # diffuse shading
    )
    # Add wireframe box around original volume
    bounds = recon_vol.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    box = pv.Box(bounds=bounds)
    plotter.add_mesh(box, color='black', style='wireframe', line_width=2)
    add_axes(plotter, xlabel='O', ylabel='L', zlabel='T', color=("red", "blue", "green"))
    # T = Top (bottom to top)
    # O = Out (out of the hilum)
    # L = left (from right to left)

    # Set the camera position and show the plot
    plotter.camera_position = 'iso'
    plotter.show()

def display_normalized_kidneys(rk_kidney, lk_kidney, voxel_size=(1.0,1.0,1.0)):
    """
    Visualize original and reconstructed 3D volumes in PyVista with correct proportions.
    """
    voxel_size = np.array(voxel_size, dtype=float)
    plotter = pv.Plotter(window_size=(800,600), shape=(1,2))
    plotter.background_color = 'white'

    # Wrap original volume
    orig_vol = pv.wrap(rk_kidney.astype(float))
    orig_vol.spacing = voxel_size
    orig_surface = orig_vol.contour(isosurfaces=[0.5])
    plotter.subplot(0,0)
    plotter.add_text(f"Right kidney", font_size=12)
    plotter.add_mesh(
        orig_surface, color='lightblue', opacity=1.0, style='surface',     
        ambient=0.1,           # ambient term
        diffuse=0.9,           # diffuse shading
    )
    # Add wireframe box around original volume
    bounds = orig_vol.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    box = pv.Box(bounds=bounds)
    plotter.add_mesh(box, color='black', style='wireframe', line_width=2)
    add_axes(plotter, xlabel='LO', ylabel='PO', zlabel='HO', color=("red", "blue", "green"))

    # Wrap reconstructed volume
    recon_vol = pv.wrap(lk_kidney.astype(float))
    recon_vol.spacing = voxel_size
    recon_surface = recon_vol.contour(isosurfaces=[0.5])
    plotter.subplot(0,1)
    plotter.add_text(f"Left kidney flipped", font_size=12)
    plotter.add_mesh(recon_surface, color='lightblue', opacity=1.0, style='surface',
        ambient=0.1,           # ambient term
        diffuse=0.9,           # diffuse shading
    )
    # Add wireframe box around original volume
    bounds = recon_vol.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    box = pv.Box(bounds=bounds)
    plotter.add_mesh(box, color='black', style='wireframe', line_width=2)
    add_axes(plotter, xlabel='LO', ylabel='PO', zlabel='HO', color=("red", "blue", "green"))

    # Set the camera position and show the plot
    plotter.camera_position = 'iso'
    plotter.show()


def compare_processed_kidneys(kidney, kidney_proc, voxel_size=(1.0,1.0,1.0)):
    """
    Visualize original and reconstructed 3D volumes in PyVista with correct proportions.
    """
    voxel_size = np.array(voxel_size, dtype=float)
    plotter = pv.Plotter(window_size=(800,600), shape=(1,2))
    plotter.background_color = 'white'

    # Wrap original volume
    orig_vol = pv.wrap(kidney.astype(float))
    orig_vol.spacing = voxel_size
    orig_surface = orig_vol.contour(isosurfaces=[0.5])
    plotter.subplot(0,0)
    plotter.add_text(f"Original", font_size=12)
    plotter.add_mesh(
        orig_surface, color='lightblue', opacity=1.0, style='surface',     
        ambient=0.1,           # ambient term
        diffuse=0.9,           # diffuse shading
    )
    # Add wireframe box around original volume
    bounds = orig_vol.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    box = pv.Box(bounds=bounds)
    plotter.add_mesh(box, color='black', style='wireframe', line_width=2)
    #add_axes(plotter, xlabel='LO', ylabel='PO', zlabel='HO', color=("red", "blue", "green"))
    add_axes(plotter, xlabel='O', ylabel='L', zlabel='T', color=("red", "blue", "green"))
    # # Force camera to look from the opposite direction
    # plotter.view_vector((-1, 0, 0))  # rotate 180° around vertical axis

    # Wrap reconstructed volume
    recon_vol = pv.wrap(kidney_proc.astype(float))
    recon_vol.spacing = voxel_size
    recon_surface = recon_vol.contour(isosurfaces=[0.5])
    plotter.subplot(0,1)
    plotter.add_text(f"Processed", font_size=12)
    plotter.add_mesh(recon_surface, color='lightblue', opacity=1.0, style='surface',
        ambient=0.1,           # ambient term
        diffuse=0.9,           # diffuse shading
    )
    # Add wireframe box around original volume
    bounds = recon_vol.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    box = pv.Box(bounds=bounds)
    plotter.add_mesh(box, color='black', style='wireframe', line_width=2)
    # add_axes(plotter, xlabel='LO', ylabel='PO', zlabel='HO', color=("red", "blue", "green"))
    add_axes(plotter, xlabel='O', ylabel='L', zlabel='T', color=("red", "blue", "green"))

    # Set the camera position and show the plot
    plotter.camera_position = 'iso'
    plotter.show()


def display_volumes_two_panel(original_volume, reconstructed_volume, original_voxel_size=(1.0,1.0,1.0), reconstructed_voxel_size=None):
    """
    Visualize original and reconstructed 3D volumes in PyVista with correct proportions.
    """
    original_voxel_size = np.array(original_voxel_size, dtype=float)
    if reconstructed_voxel_size is None:
        reconstructed_voxel_size = original_voxel_size
    plotter = pv.Plotter(window_size=(800,600), shape=(1,2))
    plotter.background_color = 'white'

    # Wrap original volume
    orig_vol = pv.wrap(original_volume.astype(float))
    orig_vol.spacing = original_voxel_size
    orig_surface = orig_vol.contour(isosurfaces=[0.5])
    plotter.subplot(0,0)
    plotter.add_text("Original", font_size=12)
    plotter.add_mesh(orig_surface, color='lightblue', opacity=1.0, style='surface')
    # Add wireframe box around original volume
    bounds = orig_vol.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    box = pv.Box(bounds=bounds)
    plotter.add_mesh(box, color='black', style='wireframe', line_width=2)
    add_axes(plotter)

    # Wrap reconstructed volume
    recon_vol = pv.wrap(reconstructed_volume.astype(float))
    recon_vol.spacing = reconstructed_voxel_size
    recon_surface = recon_vol.contour(isosurfaces=[0.5])
    plotter.subplot(0,1)
    plotter.add_text("Reconstructed", font_size=12)
    plotter.add_mesh(recon_surface, color='red', opacity=1.0, style='surface')
    # Add wireframe box around original volume
    bounds = recon_vol.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    box = pv.Box(bounds=bounds)
    plotter.add_mesh(box, color='black', style='wireframe', line_width=2)
    add_axes(plotter)

    # Set the camera position and show the plot
    plotter.camera_position = 'iso'
    plotter.show()


def display_volumes(original_volume, reconstructed_volume, original_voxel_size=(1.0,1.0,1.0), reconstructed_voxel_size=None):
    """
    Visualize original and reconstructed 3D volumes in PyVista with correct proportions.
    """
    original_voxel_size = np.array(original_voxel_size, dtype=float)
    if reconstructed_voxel_size is None:
        reconstructed_voxel_size = original_voxel_size
    plotter = pv.Plotter(window_size=(800,600))
    plotter.background_color = 'white'

    # Wrap original volume
    orig_vol = pv.wrap(original_volume.astype(float))
    orig_vol.spacing = original_voxel_size
    orig_surface = orig_vol.contour(isosurfaces=[0.5])
    plotter.add_mesh(orig_surface, color='blue', opacity=0.3, style='surface', label='Original Volume')

    # Wrap reconstructed volume
    recon_vol = pv.wrap(reconstructed_volume.astype(float))
    recon_vol.spacing = reconstructed_voxel_size
    recon_surface = recon_vol.contour(isosurfaces=[0.5])
    plotter.add_mesh(recon_surface, color='red', opacity=0.3, style='surface', label='Reconstructed Volume')

    # Add wireframe box around original volume
    bounds = orig_vol.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    box = pv.Box(bounds=bounds)
    plotter.add_mesh(box, color='black', style='wireframe', line_width=2)

    plotter.add_legend()
    plotter.add_text('3D Volume Reconstruction', font_size=20)
    plotter.camera_position = 'iso'
    plotter.show()








def display_surface(volume_recon):

    # -----------------------
    # Extract surface mesh (marching cubes)
    # -----------------------
    verts, faces, normals, _ = measure.marching_cubes(volume_recon, level=0.5)
    faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32)

    mesh = pv.PolyData(verts, faces)
    mesh_smooth = mesh.smooth(n_iter=50, relaxation_factor=0.1)

    # -----------------------
    # PyVista visualization
    # -----------------------
    plotter = pv.Plotter()
    plotter.add_mesh(mesh_smooth, color="lightblue", opacity=1.0, show_edges=False)
    plotter.add_axes()
    plotter.show_grid()
    plotter.show()
