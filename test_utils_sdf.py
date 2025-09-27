import numpy as np
import napari

from src.utils import normalize, sdf, render



def test_representation():

    voxel_size = (1.0,1.0,1.0)
    mask1 = render.ellipsoid_mask(
        (256, 256, 128), 
        voxel_sizes=voxel_size, 
        center=(50,50,0),
        radii=(45,15,15),
        rot_vec=(1,1,0),
    )
    mask2 = render.ellipsoid_mask(
        (256, 256, 128), 
        voxel_sizes=voxel_size, 
        center=(50,50,0),
        radii=(15,15,45),
        rot_vec=(1,0,1),
    )
    mask = np.logical_or(mask1, mask2)

    # Normalize
    mask_norm, params = normalize.normalize_mask(mask, voxel_size)

    # Visualize
    size = 100
    coeffs, coeffs_trunc, mask_norm_recon = sdf.compress(mask_norm, [size, size, size])

    # Visualize
    # render.display_volume(mask_norm_recon)
    render.display_volumes(mask_norm, mask_norm_recon)







# Example usage
if __name__ == '__main__':
    test_representation()