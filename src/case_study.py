import os

import dbdicom as db
import numpy as np

from utils import normalize, render, lb, sdf

DIXONS = os.path.join(os.getcwd(), 'build', 'dixon', 'stage_2_data')
SEGMENTATIONS = os.path.join(os.getcwd(), 'build', 'kidneyvol', 'stage_3_edit')

patient = '4128_031'
# patient = '4128_C08'
# patient = '4128_033'
# patient = '4128_064'
# patient = '1128_038'
# patient = '1128_081'
# patient = '4128_C25'
# patient = '4128_C22'
# patient = '7128_014'
study = 'Visit1' if 'C' in patient else 'Baseline'
task = 'kidney_masks'


def display_surface_lb():

    series = [SEGMENTATIONS, patient, study, task]
    vol = db.volume(series)
    mask = vol.values == 2
    mesh = lb.mask_to_mesh(mask) 
    coeffs, eigvals, recon_mesh = lb.process(mesh, k=100, threshold=15)
    for i, c in enumerate(coeffs):
        print(i, eigvals[i], float(np.linalg.norm(c)))
    render.visualize_surface_reconstruction(mesh, recon_mesh, opacity=(0.3,0.3))


def display_surface_sdf():
    series = [SEGMENTATIONS, patient, study, task]
    vol = db.volume(series)
    
    # Normalize and display
    mask = vol.values == 2
    mask_norm, params = normalize.normalize_mask(mask)
    render.display_volumes(mask, mask_norm)

    # Visualize
    size = 20
    coeffs_trunc, mask_norm_recon = sdf.compress(mask_norm, 3 * [size])

    # Visualize
    render.display_volumes(mask_norm, mask_norm_recon)


def display_multiple_normalized():

    for patient in ['4128_031', '4128_C08', '4128_033', '4128_064', '1128_038', '1128_081', '4128_C25', '4128_C22', '7128_014']:

        study = 'Visit1' if 'C' in patient else 'Baseline'
        series = [SEGMENTATIONS, patient, study, task]
        vol = db.volume(series)
        voxel_size = vol.spacing
        spacing_norm = 1.0
        volume_norm = 1e6
        voxel_size_norm = 3 * [spacing_norm]

        rk_mask = vol.values == 2
        rk_mask_norm, _ = normalize.normalize_kidney_mask(rk_mask, voxel_size, 'right')
        render.display_kidney_normalization(rk_mask, rk_mask_norm, voxel_size, voxel_size_norm, title='Right kidney')

        lk_mask = vol.values == 1
        lk_mask_norm, _ = normalize.normalize_kidney_mask(lk_mask, voxel_size, 'left')
        render.display_kidney_normalization(np.flip(lk_mask, 0), lk_mask_norm, voxel_size, voxel_size_norm, title='Left kidney flipped')



def display_normalized():
    series = [SEGMENTATIONS, patient, study, task]
    vol = db.volume(series)
    voxel_size = vol.spacing
    spacing_norm = 1.0
    volume_norm = 1e6
    voxel_size_norm = 3 * [spacing_norm]

    rk_mask = vol.values == 2
    rk_mask_norm, _ = normalize.normalize_kidney_mask(rk_mask, voxel_size, 'right')
    render.display_kidney_normalization(rk_mask, rk_mask_norm, title='Right kidney')

    lk_mask = vol.values == 1
    lk_mask_norm, _ = normalize.normalize_kidney_mask(lk_mask, voxel_size, 'left')
    render.display_kidney_normalization(np.flip(lk_mask, 0), lk_mask_norm, voxel_size, voxel_size_norm, title='Left kidney flipped')

    return
    # render.display_normalized_kidneys(rk_mask_norm, lk_mask_norm, voxel_size_norm)

    # Check compression

    cutoff = 64

    rk_mask_recon, _ = sdf.compress(rk_mask_norm, 3 * [cutoff])
    render.compare_processed_kidneys(rk_mask_norm, rk_mask_recon, voxel_size_norm)

    lk_mask_recon, _ = sdf.compress(lk_mask_norm, 3 * [cutoff])
    render.compare_processed_kidneys(lk_mask_norm, lk_mask_recon, voxel_size_norm)

    print('\nRight kidney volume')
    print(f"Target: {volume_norm / 1e6} Litre")
    print(f"Actual: {rk_mask_norm.sum() * spacing_norm ** 3 / 1e6} Litre")
    print(f"Compressed: {rk_mask_recon.sum() * spacing_norm ** 3 / 1e6} Litre")
    print(f"Loss: {np.around(100 * np.abs(rk_mask_recon.sum() - rk_mask_norm.sum()) / rk_mask_norm.sum(), 2) } %")
    print('\nLeft kidney volume')
    print(f"Target: {volume_norm / 1e6} Litre")
    print(f"Actual: {lk_mask_norm.sum() * spacing_norm ** 3 / 1e6} Litre")
    print(f"Compressed: {lk_mask_recon.sum() * spacing_norm ** 3 / 1e6} Litre")
    print(f"Loss: {np.around(100 * np.abs(lk_mask_recon.sum() - lk_mask_norm.sum()) / lk_mask_norm.sum(), 2) } %")


def display_normalized_npz(npz_dir):
    
    patient = '7128_014'
    study_desc = 'Baseline'
    series_desc = 'normalized_right_kidney_mask'
    filepath = os.path.join(
        npz_dir, 
        f"Patient__{patient}", 
        f"Study__1__{study_desc}",
        f"Series__1__{series_desc}.npz",
    )
    mask = np.load(filepath)['mask']
    mask = sdf.smooth_mask(mask, 3 * [32])
    render.display_volume(mask)


if __name__=='__main__':

    NPZ_DIR = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build\kidneyvol\stage_7_normalized_npz"

    # display_surface_lb()
    # display_surface_sdf()
    # display_normalized()
    # display_multiple_normalized()
    display_normalized_npz(NPZ_DIR)
