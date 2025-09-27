import numpy as np

from src.utils import zernike, render


def test_zernike_moments_3d():
    # Create a simple 3D mask (e.g., a sphere)
    mask = render.ellipsoid_mask(
        (64, 64, 32), 
        voxel_sizes=(1.0,1.0,1.0), 
        center=(0,0,0),
        radii=(20,10,5),
        rot_vec=(0,0,0),
    )

    # Compute moments up to order n_max = 4
    moments, _ = zernike.zernike_moments_3d(mask, n_max=4)
    
    print("Computed 3D Zernike Moments (n, m, l):")
    for key, value in moments.items():
        print(f"A_{key} = {value:.4f}")



def test_reconstruct_volume_3d():

    ell_1 = render.ellipsoid_mask(
        (64, 64, 64), 
        voxel_sizes=(1.0,1.0,1.0), 
        center=(0,0,0),
        radii=(30,10,10),
        rot_vec=(1,0,0),
    )
    ell_2 = render.ellipsoid_mask(
        (64, 64, 64), 
        voxel_sizes=(1.0,1.0,1.0), 
        center=(0,0,0),
        radii=(25,15,10),
        rot_vec=(0,1,1),
    )
    original_mask = np.logical_or(ell_1, ell_2)
    # render.display_volume(original_mask)

    # Compute moments up to order n_max = 4
    moments, (max_dist_val, centroid_val) = zernike.zernike_moments_3d(original_mask, n_max=20)
    
    # Reconstruct the volume
    recon = zernike.reconstruct_volume_3d(
        moments, original_mask.shape, max_dist_val, centroid_val)
    
    recon_mask = recon > np.percentile(recon, 15)
    render.display_volumes(original_mask, recon_mask)


# Example usage
if __name__ == '__main__':
    # test_zernike_moments_3d()
    test_reconstruct_volume_3d()